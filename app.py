import sys
sys.path.insert(1, 'src')
from typing import Tuple
import time
import re
import numpy as np
import cv2
from PIL import Image
from threading import Thread

# Whisper
from speechstream import StreamHandler

# LLMs
from gemma import Gemma
from llama import Llamav2
from customgpt import CustomGPT

# LLVM
from owlsimple import HootHoot


def parse_commands(text: str) -> Tuple[str, list]:
    # Remove all \n and \t
    text = text.replace("\n", "")
    text = text.replace("\t", "")

    # Replace AI with A.I.
    text = text.replace("AI", "A.I.")

    # Remove all text between * and *
    remove = re.sub(r'\*(.*?)\*', '', text)
    for r in remove:
        text = text.replace(f"*{r}*", "")

    # Extract all text between @ and @
    commands = re.findall(r'@(.*?)@', text) 
    
    # Remove all commands from text
    for command in commands:
        text = text.replace(f"@{command}@", "")
    
    return text, commands

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

        time.sleep(0.1)

def vision(
        owl: HootHoot, 
        v4l2: int
    ) -> None:
    source = cv2.VideoCapture(v4l2)
    while True:
        ret, frame = source.read()
        if ret:
            # Convert frame to pil
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            output, image = owl.predict(frame)
            # Convert image to numpy
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow("ZByHP", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    source.release()
    cv2.destroyAllWindows()

def main(
        asr: bool = False,
        cv: bool = True, 
        model: str = "gpt"
    ) -> None:

    # Initialise LLM
    llm = None
    if model == "gpt":
        llm = CustomGPT()
    elif model == "llama":
        llm = Llamav2()
    else: 
        llm = Gemma()
    
    if cv:
        # Initialise OWL
        owl = HootHoot()

        vision_thread = Thread(
            target=vision, 
            args=(
                owl,
                0
            )
        )
        vision_thread.daemon = True
        vision_thread.start()
    
    # Initialise Whisper
    if asr:
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
            text, commands = parse_commands(robot_output)
            print("Veronica: {}".format(text))
            if commands:
                print("Commands: {}".format(commands))
                owl.update_prompt(commands)
    

if __name__ == "__main__":
    main()