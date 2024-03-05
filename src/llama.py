from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Llamav2:
    def __init__(
            self,
            model_path: str = "models/llama-2-7b-chat-gptq-4bit-128g",   
            device_map: str = "auto",
            trust_remote_code: bool = False,
            revision: str = "main",  
            use_fast: bool = True,
            max_new_tokens: int = 512,
            do_sample: bool = True,
            temperature: float = 0.6,
            top_p: float = 0.95,
            top_k: int = 10,
            repetition_penalty: float = 1.2
        ) -> None:

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            revision=revision,
            attn_implementation="flash_attention_2"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=use_fast
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
    def get_prompt(self, query: str) -> str:
        prompt_template=f'''[INST] <<SYS>>
        You are an AI assistant named Veronica, with computer vision capabilities. \
        All your responses are super-short. \
        You have a camera and can perform advanced vision capabilites like object detection. \
        When asked to find or see an object, you can use the following commands: \
        @DETECT_START_X@ - detect object X where X can be objects like person, cat, dog, etc, based on the users input. Example, @DETECT_START_PERSON@ \
        @DETECT_STOP_X@ - stop detecting object X where X is the object according to user's input. Example, @DETECT_STOP_TOYS@ \
        @DETECT_STOP_ALL@ - stop detecting all objects \
        Use this command matrix as output when asked requested to detect items.
        <</SYS>>
        {query}[/INST]
        '''
        return prompt_template
    
    def exec(self, query: str) -> str:
        prompt = self.get_prompt(query)
        answer = self.pipe(prompt)[0]['generated_text']
        answer = answer.split("[/INST]")[1].strip()
        return answer