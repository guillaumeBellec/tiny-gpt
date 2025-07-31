# Tiny-gpt

This repo is a minimal pytorch code to train a GPT 2 in 10 minutes with 4x H100.  

It is built upon the nano-gpt [1] repo and more specifically the version [2]. 
One of the main difference was the integration of the hugging face Fine web datasets with Hugging face data loaders.



The run command is:  
`` torchrun --standalone --nproc_per_node=4 main.py ``

[1] https://github.com/KellerJordan/modded-nanogpt/tree/master  
[2] https://github.com/KellerJordan/modded-nanogpt/blob/master/records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt