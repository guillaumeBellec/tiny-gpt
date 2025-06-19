import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.backend_config.native import input_output_only_quint8_dtype_config
from torch.utils.data import DataLoader
import math
from einops import rearrange
from datasets import load_dataset
from transformers import AutoTokenizer, OpenAIGPTTokenizer
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
import time
from dataclasses import dataclass
import os
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
class TinyGPTConfig:
    vocab_size : int = 50304
    n_layer : int = 8
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 6 * 64

@dataclass
class GPT2Config:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

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

        if target is not None:

            # BOS_TOKEN_ID = 50256 (but also EOS... for gpt2)
            #docs = (torch.logical_and(idx == 50256, target != 50256)).cumsum(1)  # Shape: (16, 255) - keep batch dimension
            docs = (idx == 50256).cumsum(1)  # Shape: (16, 255) - keep batch dimension

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

        # Much more efficient - no manual filtering needed
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1)) #, ignore_index=-100)
        return logits, loss


def create_dataloader(tokenizer, cache_dir, batch_size=32, max_length=256, num_samples=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    print("Loading FineWeb dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        cache_dir=cache_dir,
        streaming=False
    )
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(num_samples))

    def tokenize_function(examples):
        # Don't pad during tokenization - waste of compute
        tokens = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=False
        )
        return {"input_ids": tokens["input_ids"]}

    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]

        # Find max length in batch (up to max_length)
        max_len = min(max_length, max(len(seq) for seq in input_ids))

        padded_inputs = []
        padded_targets = []

        for seq in input_ids:
            if len(seq) > max_len:
                seq = seq[:max_len]
            elif len(seq) < 2:  # Skip sequences too short
                continue

            # Pad sequence
            padded_seq = F.pad(seq, (0, max_len - len(seq)), value=tokenizer.pad_token_id)

            # Create input/target pairs
            input_seq = padded_seq[:-1]
            target_seq = padded_seq[1:]

            # Mask padded positions in targets
            target_seq = torch.where(
                input_seq == tokenizer.pad_token_id,
                torch.tensor(-100),
                target_seq
            )

            padded_inputs.append(input_seq)
            padded_targets.append(target_seq)

        return {
            'input_ids': torch.stack(padded_inputs),
            'targets': torch.stack(padded_targets)
        }

    # Cache tokenized dataset
    cache_file = os.path.join(cache_dir, f"tokenized_{num_samples}_{max_length}.arrow") if cache_dir else None

    tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Larger batches for tokenization
            remove_columns=dataset.column_names,  # Remove all original columns
            cache_file_name=cache_file,  # Cache the tokenized data
            load_from_cache_file=True,
            desc="Tokenizing"
        )

    dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=2,  # Add multiprocessing
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True  # Keep workers alive
        )

    return dataloader


def create_packed_dataloader(tokenizer, cache_dir, batch_size=32, max_length=256, num_samples=5000):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    print("Loading FineWeb dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        cache_dir=cache_dir,
        streaming=False
    )
    if num_samples is not None and num_samples> 0:
        dataset = dataset.select(range(num_samples))

    def tokenize_and_pack(examples):
        """Tokenize and concatenate all sequences into one long sequence"""


        chunk_length = max_length +1 # +1 because the collate with split next token, versus present token
        # Split into chunks of exactly max_length
        chunks = []
        current_chunk = []

        for text in examples['text']:
            # Tokenize without truncation first
            tokens = tokenizer(
                text,
                add_special_tokens=False,  # We'll add them manually
                padding=False,
                truncation=False,
                return_attention_mask=False
            )['input_ids']

            assert tokenizer.bos_token_id is not None
            assert tokenizer.eos_token_id is not None
            assert tokenizer.eos_token_id == tokenizer.bos_token_id

            # Add BOS token at start, EOS at end
            # if tokens[0] != tokenizer.bos_token_id:
            #    tokens = [tokenizer.bos_token_id] + tokens
            if tokens[-1] != tokenizer.eos_token_id:
                tokens = tokens + [tokenizer.eos_token_id]

            if len(current_chunk) + len(tokens) < chunk_length:
                current_chunk += tokens
            else:
                current_chunk = current_chunk + tokens[:chunk_length - len(current_chunk)] # rest of tokens is thrown away
                assert len(current_chunk) == chunk_length, f"chunk length is {len(current_chunk)} expected {chunk_length}"
                chunks.append(current_chunk)
                current_chunk = []

        #print([len(c) for c in chunks])
        chunk_tokens = sum([len(c) for c in chunks])
        #assert chunk_tokens > len(all_tokens) * 0.5, f"Should have kept at least 90% of tokens, got: {chunk_tokens}/{len(all_tokens)}"

        return {'input_ids': chunks}

    # Cache the packed dataset
    cache_file = os.path.join(cache_dir, f"packed_{num_samples}_{max_length}.arrow") if cache_dir else None

    print("Tokenizing and packing sequences...")
    packed_dataset = dataset.map(
        tokenize_and_pack,
        batched=True,
        batch_size=1000,  # Process in smaller batches to avoid memory issues
        remove_columns=dataset.column_names,
        cache_file_name=cache_file,
        load_from_cache_file=True,
        desc="Packing sequences"
    )

    def collate_fn(batch):
        """Simple collate function for packed sequences"""
        # All sequences are already max_length, so just stack them
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])

        # Create targets by shifting input by 1 position
        inputs = input_ids[:, :-1]  # All tokens except last
        targets = input_ids[:, 1:]  # All tokens except first

        return {
            'input_ids': inputs,
            'targets': targets
        }

    dataloader = DataLoader(
        packed_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True  # Shuffle for better training
    )

    print(f"Created packed dataloader with {len(packed_dataset)} sequences of length {max_length}")
    return dataloader

def train_model(args):
    """Train the simple GPT model"""

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7")

    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    config = TinyGPTConfig() if args.model_size == "tiny" else GPT2Config() if args.model_size == "small" else None
    model = GPT(config=config).to(args.device)
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    #if hasattr(config, "coordinate_descent_tuning"):
    #    config.coordinate_descent_tuning = True  # suggested by @Chillee

    if args.distributed:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module  # always contains the "raw" unwrapped model
    else:
        raw_model = model

    model = torch.compile(model)


    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model has {n_params:.1f}M parameters")
    if args.wandb: run.config.update({"num_params": n_params})

    # Create dataloader
    dataloader = create_packed_dataloader(tokenizer, args.data_dir,
                                          batch_size=args.batch_size,
                                          max_length=args.max_len,
                                          num_samples=args.num_samples)

    # Optimizer
    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.6, betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.9, 0.95), fused=True)
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = torch.optim.AdamW(matrix_params, lr=args.lr3, betas=(0.9, 0.95), weight_decay=args.wd, fused=True) #Muon(matrix_params, lr=0.04, momentum=0.95)
    optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.9, 0.95), fused=True)  # note that this learning rate is neither sensitive nor tuned
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    # scheduler type in 0, 1, 2
    schedulers = None
    if args.scheduler_type == "warmup": schedulers = [get_constant_schedule_with_warmup(o,num_warmup_steps=200, num_training_steps=args.training_steps) for o in optimizers]
    elif args.scheduler_type == "cosine": schedulers = [get_cosine_schedule_with_warmup(o,num_warmup_steps=200, num_training_steps=args.training_steps) for o in optimizers]


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
            input_ids = batch['input_ids'].to(args.device)
            targets = batch['targets'].to(args.device)

            # Forward pass
            assert input_ids.shape[1] == args.max_len and input_ids.shape[0] == args.batch_size,\
                f"got input tokens with shape: {input_ids.shape}"
            logits, loss = model(input_ids, targets)

            # Backward pass
            [o.zero_grad() for o in optimizers]
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            [o.step() for o in optimizers]
            if schedulers: [s.step() for s in schedulers]

            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1

            loss_mom = loss.item() if loss_mom is None else loss_mom * mom  + (1-mom) * loss.item()

            if total_steps % 50 == 0:
                elapsed = time.time() - start_time

                #s = tokenizer.decode(input_ids[0])
                #eos_count = (input_ids == 50256).int().sum().item()

                #print(f"input ids: len={input_ids.shape} eos_count={eos_count}, inputs_ids={input_ids[0,:24].detach().cpu().numpy()} s={s}")

                if args.wandb: run.log({
                    "loss_mom": loss_mom,
                    "loss": loss.item(),
                    "elapsed": elapsed,
                    "step": total_steps,
                    })

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


def generate_text(args, model, tokenizer: OpenAIGPTTokenizer, prompt="The quick brown fox", max_length=50):
    """Generate text with the trained model"""
    model.eval()

    # Tokenize prompt
    #model.cpu()
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False) #.to(device)
    # add bos token, not by default with this tokenizer.
    input_ids = torch.concat([tokenizer.bos_token_id * torch.ones_like(input_ids[:, 0:1]), input_ids], 1)
    input_ids = input_ids.to(args.device)

    assert input_ids[0,0] == tokenizer.bos_token_id
    assert input_ids[0,1] != tokenizer.bos_token_id

    with torch.no_grad():
        n_generated = 0
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids)

            # Get next token (simple greedy decoding)
            if n_generated == 0:
                logits[0, -1, 50256] -= 1e6 # mask EOS at first generation.
            next_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            n_generated += 1

            # Stop if we hit the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and return
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    import argparse
    import socket
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=1)
    arg_parser.add_argument("--max_len", type=int, default=2**16)
    arg_parser.add_argument("--num_samples", type=int, default=None) #2_000_000)
    arg_parser.add_argument("--total_steps", type=int, default=16_000)
    arg_parser.add_argument("--distributed", type=int, default=None)
    arg_parser.add_argument("--model_size", type=str, default="small")
    arg_parser.add_argument("--wandb", type=int, default=1)
    arg_parser.add_argument("--lr3", type=float, default=2e-3)
    arg_parser.add_argument("--wd", type=float, default=1e-1)
    arg_parser.add_argument("--scheduler_type", type=str, default="cosine")

    arg_parser.add_argument("--data_dir", type=str, default="/scratch/guillaume.bellec/fineweb/")

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = arg_parser.parse_args()
    args.date = date
    args.host_name = socket.gethostname()
    args.sim_name = f"{args.host_name}_{date}"
    args.log_folder = f"checkpoints/{args.sim_name}"
    args.device_count = torch.cuda.device_count()
    if args.distributed is None: args.distributed = args.device_count > 1

    if args.distributed:
        # set up DDP (distributed data parallel). torchrun sets this env variable
        assert torch.cuda.is_available()
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
        print(f"using device: {args.device}")

        args.master_process = (ddp_rank == 0)  # this process will do logging, checkpointing etc.
        if not args.master_process: args.wandb = 0 # no redundant logging plz
        args.training_steps = args.total_steps // ddp_world_size

    else:

        # Set device
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device (non distributed): {args.device}")
        args.master_process = True

        args.training_steps = args.total_steps

    if args.master_process:
        os.mkdir(args.log_folder)

    def print0(s, logonly=False):
        if args.master_process:
            with open(f"{args.log_folder}/log.txt", "a") as f:
                if not logonly:
                    print(s)
                f.write(s + '\n')


    print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
    import subprocess

    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print0(f'{result.stdout}', logonly=True)
    print0('=' * 100, logonly=True)

    if args.wandb:

        import wandb
        run = wandb.init(
            entity="bellec-tu-wien",
            project="tiny-gpt",
            name=args.sim_name,
            config=args.__dict__,
        )

    # Train the model
    model, tokenizer = train_model(args)

    if args.master_process:
        mem_allocated = torch.cuda.max_memory_allocated() // 1024 // 1024
        print0(f"peak memory consumption: {mem_allocated} MiB")
        if args.wandb: run.config.update({"cuda_mem_MB_allocated": mem_allocated})

        #
        if hasattr(model, "module"): # case of ddp
            model = model.module

        file = f"{args.log_folder}/model.pt"
        torch.save(model, file)
        print0(f"saved - {file}")

        # Generate some text
        print0("\n" + "=" * 50)
        print0("GENERATION EXAMPLES:")
        print0("=" * 50)

        prompts = [
            "The quick brown fox",
            "In a distant galaxy",
            "The art of programming"
        ]

        for prompt in prompts:
            generated = generate_text(args, model, tokenizer, prompt, max_length=30)
            print0(f"\nPrompt: '{prompt}'")
            print0(f"Generated: {generated}")

        print0("\nTraining complete! The model can now generate simple text.")