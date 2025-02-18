# vision_agent.py

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from agent import create_agent
from blip_default import ask_blip

# Boolean flags: 
# - useBLIP: if True, run BLIP at startup to get a scene caption.
# - useAgent: if True, use the multi-step agent; if False, directly use vision_tool.
useBLIP = True
useAgent = False

# Global variable to store the BLIP scene caption.
scene_caption = ""

# Global variables for OwlViT.
owlvit_processor = None
owlvit_model = None

def load_owlvit_model():
    """
    Loads the OwlViT model and processor if not already loaded.
    """
    global owlvit_processor, owlvit_model
    if owlvit_processor is None or owlvit_model is None:
        print("Loading OwlViT model and processor...")
        owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        owlvit_model.eval()
        print("OwlViT loaded.")
    return owlvit_processor, owlvit_model

def process_image_with_owlvit(image_path: str, threshold: float = 0.2, queries: list = None):
    """
    Processes the image with OwlViT using the provided queries.
    Lower threshold to allow more detections.
    Returns the annotated image and detection results.
    """
    processor, model = load_owlvit_model()
    image = Image.open(image_path).convert("RGB")
    
    # Use the queries provided (expecting a list of one or more keywords)
    if queries is None or len(queries) == 0:
        return image, {}  # If no queries, return image and empty results.
    
    inputs = processor(text=queries, images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    
    # Use "phrases" if available; otherwise, convert labels safely.
    if "phrases" in results:
        phrases = results["phrases"]
    else:
        phrases = []
        for label in results["labels"]:
            label_id = label.item()
            phrase = owlvit_model.config.id2label.get(label_id, f"Unknown({label_id})")
            phrases.append(phrase)
    
    # Annotate the image.
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None

    for score, phrase, box in zip(results["scores"], phrases, results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{phrase}: {round(score.item(), 3)}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), label_text, fill="red", font=font)
    
    return image, results

def ask_owlvit(query: str) -> str:
    """
    Uses OwlViT to answer a query about the image.
    For a specific query (e.g., "mug"), the query is used directly.
    For a general query like "what else", one might later incorporate candidate extraction from BLIP.
    Returns a textual summary of detections and saves the annotated image.
    """
    here = os.path.dirname(__file__)
    images_dir = os.path.join(here, "Images")
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    if not image_files:
        return "[OwlViT] No image found in ./Images folder."
    image_path = os.path.join(images_dir, image_files[0])
    
    # For now, use the provided query directly.
    queries = [query]
    
    processed_image, results = process_image_with_owlvit(image_path, threshold=0.2, queries=queries)
    processed_path = os.path.join(here, "owlvit_processed.jpg")
    processed_image.save(processed_path)
    
    if "phrases" in results:
        phrases = results["phrases"]
    else:
        phrases = []
        for label in results.get("labels", []):
            label_id = label.item()
            phrase = owlvit_model.config.id2label.get(label_id, f"Unknown({label_id})")
            phrases.append(phrase)
    
    summary = f"Detection for query '{query}':\n"
    for score, phrase, box in zip(results.get("scores", []), phrases, results.get("boxes", [])):
        summary += f"{phrase}: score {round(score.item(), 3)}, box {[round(i,2) for i in box.tolist()]}\n"
    summary += f"\nAnnotated image saved as '{processed_path}'."
    return summary

def vision_tool(query: str) -> str:
    """
    Combined vision tool.
      - If the query is empty and useBLIP is True, call BLIP (ask_blip) with an empty string to get a scene caption.
      - Otherwise, call ask_owlvit with the provided query.
    """
    if query.strip() == "":
        if useBLIP:
            return ask_blip("")
        else:
            return "Please provide a specific query."
    else:
        return ask_owlvit(query)

def main():
    global scene_caption
    print("=== Vision Agent Demo ===")
    
    # Run BLIP at startup if useBLIP is enabled.
    if useBLIP:
        scene_caption = ask_blip("")
        print("Scene Caption from BLIP:")
        print(scene_caption)
    
    # Optionally, display an initial annotated image from OwlViT (using a dummy query to show the image).
    here = os.path.dirname(__file__)
    images_dir = os.path.join(here, "Images")
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    if image_files:
        image_path = os.path.join(images_dir, image_files[0])
        annotated_image, _ = process_image_with_owlvit(image_path, queries=["dummy"])
        annotated_image.show()
        print("Annotated image displayed (also saved as 'owlvit_processed.jpg').")
    else:
        print("No image found in ./Images folder.")
    
    # System prompt for the agent.
    system_prompt = (
        "You are a vision agent. At startup, you received a scene caption from BLIP describing the image. "
        "When the user asks a question, extract only the key object(s) from the query (for example, 'mug' from 'Where is the mug?'). "
        "Immediately call VisionTool with only that keyword (using the format VisionTool(\"keyword\")) and provide the final answer "
        "in the format: Final Answer: <answer>. Do not include any additional chain-of-thought or commentary."
    )
    
    if useAgent:
        agent = create_agent(vision_tool, system_prompt)
        print("\nNow you can ask questions about the scene. Type 'exit' to quit.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("[Agent] Bye!")
                break
            response = agent.run(user_input)
            print(f"\nResponse: {response}\n")
    else:
        print("\nAgent mode is off. You can now directly type queries.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Exiting.")
                break
            response = vision_tool(user_input)
            print(f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()
