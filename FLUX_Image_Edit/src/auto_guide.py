import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME_QUANTIZED = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

OBJECTIVE = "Make the car a Toyota"
DESCRIPTION_PROMPT = (
    f'Generate a 300 word exhaustive description of everything you see on the image. Focus on small details, such as license plates, windows, mirrors, etc. '
)
CONVERTER_PROMPT = (
    f'You are given a description of an image and your task is to change it following the objective prompt "{{obj}}". '
    f'The description is the following: "{{desc}}". '
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PromptEditor:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",               # or remove this if you want `.to(device)` instead
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def process_inputs(self, image, question):
          
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question}
                ] 
                if image else 
                [
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self.processor(
            text=[text],
            images=image,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to("cuda")
        
    def generate(self, image, question):

        inputs = self.process_inputs(image, question)
        output_ids = self.model.generate(**inputs, max_new_tokens=512)

        gen_ids = [
            out_ids[ inputs["input_ids"].shape[-1]: ]
            for out_ids in output_ids
        ]

        answer = self.processor.batch_decode(gen_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return answer[0]
    
    def edit(self, original_description, objective_prompt):
        """
        This should only use the generated description and turn it 
        into a different description following the objective
        """
        editing_prompt = CONVERTER_PROMPT.format(desc=original_description, obj=objective_prompt)
        return self.generate(None, editing_prompt)
    
        
# Example:
# python auto_guide.py --image ./test_pic.png --question "What is the object on the image?"
# No question
# python auto_guide.py --image ./test_pic.png

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="path to image to ask about"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="question to ask about the image",
        default=None
    )
    args = parser.parse_args()
    editor = PromptEditor()
    pil_img = Image.open(args.image).convert("RGB")
    
    if args.question == None:
        args.question = DESCRIPTION_PROMPT
    
    image_description = editor.generate(pil_img, args.question)
    
    edited_description = editor.edit(image_description, OBJECTIVE)
    print(image_description, "\n****************************************\n", edited_description)
