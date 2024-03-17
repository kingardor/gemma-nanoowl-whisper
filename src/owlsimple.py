from typing import Tuple, List
import os
import PIL.Image

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

class HootHoot:
    def __init__(
            self, 
            model: str = "google/owlvit-base-patch32",
            onnx_path: str = "models/nanoowl/owlvit_image_encoder_patch32.onnx",
            image_encoder_engine: str = "engines/owlvit_image_encoder_patch32.engine"
    ) -> None:
        
        self.new_prompt = ['person']
        self.thresholds = [0.1]
        
        self.owl_predictor = OwlPredictor(
            model_name=model,
            device="cuda"
        )

        if not os.path.exists(onnx_path):
            print("Creating onnx model...")
            os.makedirs(image_encoder_engine.split('/')[0], exist_ok=True)
            self.owl_predictor.export_image_encoder_onnx(
                onnx_path
            )
        else:
            print("Onnx model already exists...")
        
        if not os.path.exists(image_encoder_engine):
            os.makedirs(image_encoder_engine.split('/')[0], exist_ok=True)
            print("Creating image encoder engine...")
            self.owl_predictor.build_image_encoder_engine(
                image_encoder_engine,
                fp16_mode=True,
                onnx_path=onnx_path,
                onnx_opset=17
            )

        self.predictor = OwlPredictor(
                model_name=model,
                device="cuda",
                image_encoder_engine=image_encoder_engine
        )
        
        self.text_encodings = self.predictor.encode_text(self.new_prompt)
    
    def update_prompt(self, commands: List) -> None:
        for command in commands:
            if "DETECT_STOP_ALL" in command:
                self.new_prompt = ['person']
            elif "DETECT_START" in command:
                command = command.replace("DETECT_START_", "")
                command = command.replace("_", " ")
                command = command.lower()
                self.new_prompt.append(command)
            elif "DETECT_STOP" in command:
                command = command.replace("DETECT_STOP_", "")
                command = command.replace("_", " ")
                command = command.lower()
                if command in self.new_prompt:
                    self.new_prompt.remove(command)

        self.text_encodings = self.predictor.encode_text(self.new_prompt)        
        if len(self.new_prompt) == 0:
            self.thresholds = []
        else:
            self.thresholds = [0.1] * len(self.new_prompt)
        print('Current LLVM Promt: {}'.format(self.new_prompt))

    def predict(
            self, 
            image: PIL.Image.Image
        ) -> Tuple[dict, PIL.Image.Image]:

        output = self.predictor.predict(
            image=image, 
            text=self.new_prompt,
            text_encodings=self.text_encodings,
            threshold=self.thresholds,
            pad_square=False
        )

        image = draw_owl_output(image, output, text=self.new_prompt, draw_text=True)

        return output, image