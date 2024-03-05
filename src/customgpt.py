from openai import OpenAI
from keys import OPENAI_API_KEY

class CustomGPT:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.system_prompt = "You are an AI assistant named Veronica.  \
            You are helpful and polite with a touch of sass. \
            You have advanced response and sensory vision capabilities. \
            Along with your knowledge base, you have the capability to issue detection commands. \
            Only if the request is to detect or find objects, use the following commnd matrix to respond: \
             - @DETECT_START_X@ - detect object X where X can be objects like person, cat, dog, etc, based on the users input. Example, @DETECT_START_PERSON@ \
             - @DETECT_STOP_X@ - stop detecting the relevant object. Example, @DETECT_STOP_TOY@ \
             - @DETECT_STOP_ALL@ - stop detecting all objects \
            You cannot combine START and STOP commands. \
            If your response consists of a command, do not speak anything else. This is non-negotiable."
        
        self.messages = [
            {
                "role": "system", 
                "content": self.system_prompt
            } 
        ]
    
    def exec(self, query: str) -> str:
        reply = ''
        if query:
            self.messages.append( 
                {"role": "user", "content": query}, 
            ) 
            chat = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                temperature=1.0,
                messages=self.messages) 
            
            reply = chat.choices[0].message.content 
            self.messages.append({"role": "assistant", "content": reply})
        return reply

    def exec_gradio(self, query: str, history: list) -> str:
        if isinstance(query, tuple):
            reply = "Sorry, this is not implemented yet."
        else:
            self.messages.append( 
                {"role": "user", "content": query}, 
            ) 
            chat = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                temperature=0.5,
                messages=self.messages) 
            
            reply = chat.choices[0].message.content 
            self.messages.append({"role": "assistant", "content": reply})
        
        return reply