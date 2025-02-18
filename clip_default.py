# clip_default.py

import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from agent import create_agent

# Set to True to enable multi-step reasoning with our agent,
# or False to run the single-step approach directly.
useAgent = False

# Default CLIP model to use; change to try different models.
clip_model_name = "openai/clip-vit-base-patch32"  # medium size model

def ask_clip(query: str) -> str:
    """
    CLIP approach: computes a similarity score between a text query and the single image found in the Images folder.
    Returns a string like: "[CLIP] Query='<query>', similarity=0.xxxx"
    """
    here = os.path.dirname(__file__)
    img_dir = os.path.join(here, "Images")
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    if not image_files:
        return "[CLIP] No image found in ./Images folder."
    image_path = os.path.join(img_dir, image_files[0])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()
    
    # Process text
    text_inputs = processor(text=[query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    
    # Process image
    image_pil = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        image_feats = model.get_image_features(**image_inputs)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    
    sim = torch.matmul(text_feats, image_feats.T).item()
    return f"[CLIP] Query='{query}', similarity={sim:.4f}"

def main():
    print("=== CLIP Demo with Agent Option ===")
    print("Place exactly one image in the ./Images folder.")
    print("Type 'exit' to quit.\n")

    # Define a system prompt that the agent will use for multi-step reasoning.
    system_prompt = (
        "You have a VisionTool that computes text-to-image similarity. "
        "When a user asks about the image, call the VisionTool and interpret its result. "
        "Provide a short conclusion and then finalize your answer."
    )

    if useAgent:
        agent = create_agent(ask_clip, system_prompt)

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("[Agent] Bye!")
            break

        if useAgent:
            response = agent.run(user_input)
        else:
            response = ask_clip(user_input)
        print(f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()
