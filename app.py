import sys
sys.path.insert(1, 'src')
import time
from threading import Thread
from speechstream import StreamHandler
from gemma import Gemma

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

def main():

    # Initialise Gemma
    llm = Gemma()

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

if __name__ == "__main__":
    main()