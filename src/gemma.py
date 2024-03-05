import torch
from keys import GEMMA_API_KEY
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma:
    def __init__(self, model_name: str = "google/gemma-2b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=GEMMA_API_KEY
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            token=GEMMA_API_KEY
        )

    def exec(self, input_text: str) -> str:
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(**input_ids)
        return self.tokenizer.decode(outputs[0])
        