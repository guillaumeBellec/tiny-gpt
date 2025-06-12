import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from einops import rearrange
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
import time
from dataclasses import dataclass
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1, block_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1, block_mask)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 8 #12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 6 * 64 #768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target=None):

        docs = (idx == 50256).cumsum(1)  # Shape: (16, 255) - keep batch dimension

        if target is not None:
            def document_causal_mask(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                document_mask = docs[b, q_idx] == docs[b, kv_idx]  # Index into specific batch
                window_mask = q_idx - kv_idx < 1024
                return causal_mask & document_mask & window_mask

            S = idx.shape[1]  # Sequence length: 255
            block_mask = create_block_mask(document_causal_mask, idx.shape[0], None, S, S, device="cuda", _compile=True)
        else:
            block_mask = None

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15) # @Grad62304977
        logits = logits.float()

        if target is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss


def create_dataloader(tokenizer, batch_size=32, max_length=256, num_samples=5000):
    """Create a dataloader from FineWeb dataset"""

    # Set up cache directory
    cache_dir = "/home/guillaume/_Programming/data"
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading FineWeb dataset...")
    # Load FineWeb sample-10BT (smallest subset, ~10B tokens)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",  # This is the 10B token sample
        split="train",
        cache_dir=cache_dir,
        streaming=True  # Use streaming for large datasets
    )

    # Take only a subset for quick training
    dataset = dataset.take(num_samples)

    def tokenize_function(examples):
        # Tokenize the text - FineWeb uses 'text' column
        tokens = tokenizer(
            examples['text'],
            padding=True, # TODO: ideally, no need to pad. Rather append sequences after one another.
            truncation=True,
            max_length=max_length,
            add_special_tokens=True  # Ensure proper special tokens
        )
        return {"input_ids": tokens["input_ids"]}

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "id", "dump", "url", "file_path", "language", "language_score", "token_count", "score"]
        # Remove FineWeb columns
    )

    def collate_fn(batch):
        # Handle unknown tokens by replacing with UNK token
        input_ids = [item['input_ids'] for item in batch]

        # Get UNK token ID (fallback to 0 if not available)
        unk_token_id = getattr(tokenizer, 'unk_token_id', 0)
        if unk_token_id is None:
            unk_token_id = 0

        max_len = min(max_length, max(len(seq) for seq in input_ids))

        padded_inputs = []
        padded_targets = []

        for seq in input_ids:
            seq = torch.tensor(seq)

            if len(seq) > max_len:
                seq = seq[:max_len]

            # Create input and target (shifted by 1)
            padded_seq = F.pad(seq, (0, max_len - len(seq)), value=tokenizer.pad_token_id)

            input_seq = padded_seq[:-1]
            target_seq = padded_seq[1:]

            # Replace -1 tokens with UNK token and mask padded tokens
            #target_seq = torch.where(target_seq == -1, unk_token_id, target_seq)
            #target_seq = torch.where(
            #    input_seq == tokenizer.pad_token_id,
            #    torch.tensor(-100),  # Use -100 for ignored tokens in loss
            #    target_seq
            #)

            padded_inputs.append(input_seq)
            padded_targets.append(target_seq)

        return {
            'input_ids': torch.stack(padded_inputs),
            'targets': torch.stack(padded_targets)
        }

    # Convert streaming dataset to iterable for DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return dataloader


def train_model(args):
    """Train the simple GPT model"""

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model = GPT(config=GPTConfig()).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create dataloader
    dataloader = create_dataloader(tokenizer, batch_size=args.bach_size, max_length=args.max_len, num_samples=args.num_samples)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,  # Fast warmup
        #num_training_steps=args.training_steps,  # Total training steps
        last_epoch=-1
    )

    # Training loop
    model.train()
    total_steps = 0
    start_time = time.time()

    print("Starting training...")

    loss_mom = None
    mom = 0.99

    for epoch in range(3):  # Small number of epochs for quick training
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            # Forward pass
            logits, loss = model(input_ids, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1

            loss_mom = loss.item() if loss_mom is None else loss_mom * mom  + (1-mom) * loss.item()

            if total_steps % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Step {total_steps}, Loss: {loss_mom:.4f}, Time: {elapsed:.1f}s")

            if total_steps > args.training_steps:
                break



        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        if total_steps > args.training_steps:
            break

    print("Training completed!")
    return model, tokenizer


def generate_text(model, tokenizer, prompt="The quick brown fox", max_length=50):
    """Generate text with the trained model"""
    model.eval()

    # Tokenize prompt
    model.cpu()
    input_ids = tokenizer.encode(prompt, return_tensors="pt") #.to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids)

            # Get next token (simple greedy decoding)
            next_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            # Stop if we hit the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and return
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--bach_size", type=int, default=4)
    arg_parser.add_argument("--max_len", type=int, default=1024)
    arg_parser.add_argument("--num_samples", type=int, default=50_000)
    arg_parser.add_argument("--training_steps", type=int, default=10_000)
    args = arg_parser.parse_args()


    # Train the model
    model, tokenizer = train_model(args)

    # Generate some text
    print("\n" + "=" * 50)
    print("GENERATION EXAMPLES:")
    print("=" * 50)

    prompts = [
        "The quick brown fox",
        "In a distant galaxy",
        "The art of programming"
    ]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=30)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")

    print("\nTraining complete! The model can now generate simple text.")