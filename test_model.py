import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

# Define the class names
CLASS_NAMES = [
    'dog', 'cat', 'horse', 'spider', 'butterfly', 
    'chicken', 'sheep', 'cow', 'squirrel', 'elephant'
]

# Map Italian folder names to English class names
ITALIAN_TO_ENGLISH = {
    'cane': 'dog',
    'gatto': 'cat',
    'cavallo': 'horse',
    'ragno': 'spider',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'pecora': 'sheep',
    'mucca': 'cow',
    'scoiattolo': 'squirrel',
    'elefante': 'elephant'
}

# Global device variable to ensure consistency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

def load_model(model_path):
    """Load the TorchScript model"""
    try:
        model = torch.jit.load(model_path, map_location=DEVICE)
        model.eval()
        print(f"Model loaded successfully on {DEVICE}!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def preprocess_image(image_path):
    """Preprocess the image using the same transformations from training"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
        
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        
        # Create the transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transforms
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Move tensor to the same device as the model
        img_tensor = img_tensor.to(DEVICE)
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        sys.exit(1)

def predict(model, img_tensor):
    """Get prediction from the model"""
    try:
        with torch.no_grad():
            # Ensure the tensor is on the correct device
            if img_tensor.device != next(model.parameters()).device:
                print(f"Warning: Moving input tensor from {img_tensor.device} to {next(model.parameters()).device}")
                img_tensor = img_tensor.to(next(model.parameters()).device)
                
            outputs = model(img_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            return probabilities.cpu().numpy()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Input tensor device: {img_tensor.device}")
        try:
            model_device = next(model.parameters()).device
            print(f"Model device: {model_device}")
        except:
            print("Could not determine model device")
        sys.exit(1)

def visualize_results(image_path, probabilities, class_names, top_class, expected_class=None):
    """Create and save a visualization of the image and prediction results"""
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Animal Classification Results', fontsize=16)
    
    # Display the image
    img = Image.open(image_path).convert('RGB')
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Sort probabilities for visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_labels = [class_names[i] for i in sorted_indices]
    
    # Only show top 5 predictions in the chart for clarity
    top_k = 5
    if len(sorted_probs) > top_k:
        sorted_probs = sorted_probs[:top_k]
        sorted_labels = sorted_labels[:top_k]
    
    # Create horizontal bar chart
    bars = ax2.barh(sorted_labels, sorted_probs * 100)
    
    # Add percentage labels to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center')
    
    # Highlight the top prediction
    for i, label in enumerate(sorted_labels):
        if label == top_class:
            bars[i].set_color('green')
            break
    
    # Add chart details
    ax2.set_title('Top Predictions')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_xlim(0, 100)  # Set x-axis from 0-100%
    
    # Add a note if we know the expected class
    if expected_class:
        match_text = f"Expected: {expected_class}"
        if expected_class == top_class:
            match_text += " ✅"
        else:
            match_text += " ❌"
        fig.text(0.5, 0.02, match_text, ha='center', fontsize=12)
    
    # Make layout tight
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"prediction_result_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show plot (will be displayed in VSCode's plot viewer)
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the animal classification model on a single image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--model", default="model_scripted_efficientnet_lr0.001_aughigh.pt", 
                        help="Path to the model file")
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Preprocess the image
    img_tensor = preprocess_image(args.image_path)
    
    # Get predictions
    probabilities = predict(model, img_tensor)
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 40)
    
    # Get the class with the highest probability
    top_prediction = np.argmax(probabilities)
    top_class = CLASS_NAMES[top_prediction]
    top_prob = probabilities[top_prediction]
    
    print(f"Top prediction: {top_class.upper()} with {top_prob*100:.2f}% confidence\n")
    
    # Print all class probabilities
    print("All class probabilities:")
    print("-" * 40)
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        class_name = CLASS_NAMES[idx]
        prob = probabilities[idx]
        print(f"{i+1}. {class_name.ljust(10)}: {prob*100:.2f}%")
    
    # Try to determine if the input is from the dataset
    image_path = args.image_path
    for italian, english in ITALIAN_TO_ENGLISH.items():
        if italian in image_path:
            print(f"\nNote: This image appears to be from the '{italian}' folder, which corresponds to '{english}'")
            if english == top_class:
                print("✅ The model's prediction matches the expected class!")
            else:
                print(f"❌ The model predicted '{top_class}' instead of the expected '{english}'")
            
            # Visualize the results with the expected class
            visualize_results(image_path, probabilities, CLASS_NAMES, top_class, expected_class=english)
            break
    else:
        # If not from the dataset, just visualize without expected class
        visualize_results(image_path, probabilities, CLASS_NAMES, top_class)

if __name__ == "__main__":
    main()

