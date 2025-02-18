# blip_default.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from agent import create_agent

# Set to True to enable multi-step reasoning with our agent,
# or False to run the single-step approach directly.
useAgent = False

# Use the smallest available BLIP-2 model.
blip_model_name = "salesforce/blip2-opt-2.7b"

# Global variables to cache the loaded model and processor.
processor = None
model = None

def download_model_if_needed(repo_id: str) -> str:
    """
    Downloads the model repository to a local /models folder if not already present.
    Returns the local directory path to the model.
    """
    root_dir = os.path.dirname(__file__)
    models_folder = os.path.join(root_dir, "models")
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    model_local_dir = os.path.join(models_folder, repo_id.replace("/", "_"))
    
    if not os.path.exists(model_local_dir):
        print(f"Downloading model '{repo_id}' to {model_local_dir} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_local_dir)
    else:
        print(f"Model '{repo_id}' already downloaded in {model_local_dir}.")
    return model_local_dir

def load_blip_model():
    """
    Loads the processor and model from disk only once.
    """
    global processor, model
    if processor is None or model is None:
        model_dir = download_model_if_needed(blip_model_name)
        print("Loading processor and model...")
        processor = AutoProcessor.from_pretrained(model_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.eval()
        print("Processor and model loaded.")
    # Else, they're already loaded.

def ask_blip(query: str) -> str:
    """
    BLIP-2 VQA approach: processes a question about the image found in the Images folder.
    Returns a string like: "[BLIP2] Question='<query>', Answer='<answer>'"
    """
    here = os.path.dirname(__file__)
    images_dir = os.path.join(here, "Images")
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    if not image_files:
        return "[BLIP2] No image found in ./Images folder."
    
    image_path = os.path.join(images_dir, image_files[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and processor if not already loaded.
    load_blip_model()
    
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, text=query, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)
    answer = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return f"[BLIP2] Question='{query}', Answer='{answer}'"

def main():
    print("=== BLIP-2 Demo with Agent Option ===")
    print("Place exactly one image in the ./Images folder.")
    print("Type 'exit' to quit.\n")

    system_prompt = (
        "You have a VisionTool that performs visual question answering using BLIP-2. "
        "When the user asks a question about the image, call the VisionTool and provide a detailed answer. "
        "If further clarification is needed, you may ask follow-up questions before finalizing your response."
    )

    # Load the model once at startup.
    load_blip_model()

    if useAgent:
        agent = create_agent(ask_blip, system_prompt)

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("[Agent] Bye!")
            break

        if useAgent:
            response = agent.run(user_input)
        else:
            response = ask_blip(user_input)
        print(f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()
