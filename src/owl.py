from typing import Tuple, List
import os
import PIL.Image

from nanoowl.tree_predictor import OwlPredictor, TreePredictor, Tree
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_drawing import draw_owl_output

class HootHoot:
    def __init__(
            self, 
            model: str = "google/owlvit-base-patch32",
            onnx_path: str = "models/nanoowl/owlvit_image_encoder_patch32.onnx",
            image_encoder_engine: str = "engines/owlvit_image_encoder_patch32.engine"
    ) -> None:
        
        self.old_prompt = []
        self.new_prompt = []
        self.tree = None
        self.clip_text_encodings = None
        self.owl_text_encodings = None
    
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

        self.predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                model_name=model,
                device="cuda",
                image_encoder_engine=image_encoder_engine
            )
        )

        self.update_encodings(self.new_prompt)
    
    def update_prompt(self, commands: List) -> None:
        for command in commands:
            if "DETECT_STOP_ALL" in command:
                self.new_prompt = []
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
        
        self.update_encodings(self.new_prompt)
                            
    def update_encodings(self, prompt: List) -> None:
        p = '[]'
        if len(prompt) > 0:
            p = '[ '
            for i in prompt:
                p += i + ", "
            p = p[:-2]
            p += ' ]'
            print("LLVM Prompt: {}".format(p))
        self.tree = Tree.from_prompt(p)
        self.clip_text_encodings = self.predictor.encode_clip_text(self.tree)
        self.owl_text_encodings = self.predictor.encode_owl_text(self.tree)
    
    def predict(
            self, 
            image: PIL.Image.Image, 
            threshold: float = 0.1
        ) -> Tuple[dict, PIL.Image.Image]:

        output = self.predictor.predict(
            image=image, 
            tree=self.tree,
            clip_text_encodings=self.clip_text_encodings,
            owl_text_encodings=self.owl_text_encodings,
            threshold=threshold
        )

        image = draw_tree_output(image, output, tree=self.tree, draw_text=True)

        return output, image