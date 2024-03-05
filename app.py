import sys
sys.path.insert(1, 'src')
import time
from threading import Thread
from speechstream import StreamHandler
from gemma import Gemma
from llama import Llamav2
from customgpt import CustomGPT

def conversation(
        handler: StreamHandler,
        llm: Gemma
    ) -> None:

    speaking_complete = True
    while True:
        # Listening emote
        if handler.speaking:
            speaking_complete = False
        else:
            speaking_complete = True
            if not speaking_complete:
                pass
        if not isinstance(handler.stt_result, type(None)):
            user_input = handler.stt_result
            handler.stt_result = None
            print("Me: {}".format(user_input))
            robot_output = llm.exec(user_input)
            print("Veronica: {}".format(robot_output))

        time.sleep(0.25)

def main(asr: bool = False, model: str = "gpt"):

    # Initialise LLM
    llm = None
    if model == "gpt":
        llm = CustomGPT()
    elif model == "llama":
        llm = Llamav2()
    else: 
        llm = Gemma()

    if asr:
        # Initialise Whisper
        handler = StreamHandler()

        conversation_thread = Thread(
            target=conversation, 
            args=(
                    handler,
                    llm
                )
            )
        conversation_thread.daemon = True
        conversation_thread.start()

        while True:
            time.sleep(1)
    
    else:
        while True:
            user_input = input("Me: ")
            robot_output = llm.exec(user_input)
            print("Veronica: {}".format(robot_output))
    

if __name__ == "__main__":
    main()