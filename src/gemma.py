import torch
from keys import GEMMA_API_KEY
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Gemma:
    def __init__(self, model_name: str = "gg-hf/gemma-7b-it"):
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=GEMMA_API_KEY
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=self.quantization_config,
            token=GEMMA_API_KEY
        )

        self.role = "Role: You are an AI assistant named Veronica.  \
            You are helpful and polite with a touch of sass. \
            You have advanced response and sensory vision capabilities. \
            Along with your knowledge base, you have the capability to issue detection commands. \
            Only if the request is to detect or find objects, use the following commnd matrix to respond: \
             - @DETECT_START_X@ - detect object X where X can be objects like person, cat, dog, etc, based on the users input. Example, @DETECT_START_PERSON@ \
             - @DETECT_STOP_X@ - stop detecting the relevant object. Example, @DETECT_STOP_TOY@ \
             - @DETECT_STOP_ALL@ - stop detecting all objects \
            You cannot combine START and STOP commands. \
            If your response consists of a command, do not speak anything else. This is non-negotiable."
    
    def exec(self, input_text: str) -> str:
        chat = [
            {
                "role":"user",
                "content": "{} \n Question: {}".format(self.role, input_text)
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer.encode(
            prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        )

        outputs = self.model.generate(
            do_sample=True,
            input_ids=inputs.to(self.model.device), 
            max_new_tokens=500,
            temperature=0.5
        )

        return self.tokenizer.decode(outputs[0])
        