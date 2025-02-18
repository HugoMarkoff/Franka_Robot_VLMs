# owlvit_demo.py

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from agent import create_agent

# Set to True to enable multi-step reasoning with our agent,
# or False to simply use the vision tool function.
useAgent = True

# Global variables to cache the OwlViT model and processor.
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

def process_image_with_owlvit(image_path: str, threshold: float = 0.3):
    """
    Processes the image with OwlViT to obtain object detection results,
    draws bounding boxes and labels on the image, and returns both the processed image and detection results.
    """
    processor, model = load_owlvit_model()
    
    image = Image.open(image_path).convert("RGB")
    
    # Define a list of text queries (common object categories)
    queries = [
        "a person", "a bicycle", "a car", "a motorcycle", "a bus", "a truck",
        "a dog", "a cat", "a chair", "a table", "a tree", "a building", "a road"
    ]
    
    inputs = processor(text=queries, images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    # Use the new grounded post-processing method.
    results = processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    
    # Check if "phrases" exist; otherwise, convert "labels" safely.
    if "phrases" in results:
        phrases = results["phrases"]
    else:
        phrases = []
        for label in results["labels"]:
            label_id = label.item()
            # Safely get the label, fallback if missing.
            phrase = owlvit_model.config.id2label.get(label_id, f"Unknown({label_id})")
            phrases.append(phrase)
    
    # Draw bounding boxes and predicted labels on the image.
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

def vision_tool(query: str, detection_results) -> str:
    """
    Returns a textual summary of the detection results.
    In a full implementation, the query could be used to filter or refine the output.
    """
    if "phrases" in detection_results:
        phrases = detection_results["phrases"]
    else:
        phrases = []
        for label in detection_results["labels"]:
            label_id = label.item()
            phrase = owlvit_model.config.id2label.get(label_id, f"Unknown({label_id})")
            phrases.append(phrase)
        
    summary = "Detected objects:\n"
    for score, phrase, box in zip(detection_results["scores"], phrases, detection_results["boxes"]):
        summary += f"{phrase}: score {round(score.item(), 3)}, box {[round(i,2) for i in box.tolist()]}\n"
    return summary

def main():
    print("=== OwlViT Demo ===")
    print("Processing image in the ./Images folder with OwlViT...")

    here = os.path.dirname(__file__)
    images_dir = os.path.join(here, "Images")
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    if not image_files:
        print("No image found in ./Images folder.")
        return
    
    image_path = os.path.join(images_dir, image_files[0])
    
    processed_image, detection_results = process_image_with_owlvit(image_path)
    processed_image.show()  # Opens the image in your default viewer.
    processed_path = os.path.join(here, "owlvit_processed.jpg")
    processed_image.save(processed_path)
    print(f"Processed image saved as '{processed_path}'.")
    
    system_prompt = (
        "You have a VisionTool that provides object detection information from an image processed by OwlViT. "
        "Use the provided detection results to answer questions about object placement and content."
    )
    
    def tool_func(query: str) -> str:
        return vision_tool(query, detection_results)
    
    if useAgent:
        agent = create_agent(tool_func, system_prompt)
    
    print("\nNow you can ask questions about the image. Type 'exit' to quit.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("[Agent] Bye!")
            break
        if useAgent:
            response = agent.run(user_input)
        else:
            response = tool_func(user_input)
        print(f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()
