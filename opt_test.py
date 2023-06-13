import deepspeed
import os
import torch
from deepspeed.runtime.utils import memory_status
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from util import load_wikitext

def main():
    
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    model_name = "facebook/opt-13b"

    model = AutoModelForCausalLM.from_pretrained(model_name)
        
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": { "lr": 5e-5 }
        },
        "fp16": { "enabled": True },
        "zero_optimization": { "stage": 3 }
    }

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )
    model_engine.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    train_dataset = load_wikitext(tokenizer, collator, max_length=512).select(range(128))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size)
    )

    device = torch.device("cuda", rank)  
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_engine(input_ids, labels=labels)
        
        model_engine.backward(outputs.loss)
        model_engine.step()

        memory_status("Memory stats after training step")

    return

if __name__ == "__main__":
    main()