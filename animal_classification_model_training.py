# Importación de librerías necesarias
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
import mlflow
import mlflow.pytorch
from datetime import datetime

# Configuración de reproducibilidad y dispositivo
torch.manual_seed(42)
np.random.seed(42)

# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
print(f"Dispositivo utilizado: {device}")

# Configuración inicial del entrenamiento
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # Reducido para optimizar tiempo manteniendo precisión
MODEL_PATH = 'animal_classifier_model.pth'
DATASET_PATH = 'animals10'

# Definición de las clases de animales
CLASS_NAMES = [
    'dog', 'cat', 'horse', 'spider', 'butterfly', 
    'chicken', 'sheep', 'cow', 'squirrel', 'elephant'
]


def create_model(num_classes, experiment_variant='default'):
    """Crear y configurar el modelo de clasificación con diferentes arquitecturas"""
    if experiment_variant == 'resnet18':
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        model_name = "ResNet18"
    elif experiment_variant == 'resnet34':
        base_model = models.resnet34(weights='IMAGENET1K_V1')
        model_name = "ResNet34"
    elif experiment_variant == 'resnet50':
        base_model = models.resnet50(weights='IMAGENET1K_V1')
        model_name = "ResNet50"
    elif experiment_variant == 'mobilenet':
        base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model_name = "MobileNetV2"
    elif experiment_variant == 'efficientnet':
        base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model_name = "EfficientNet-B0"
    elif experiment_variant == 'densenet':
        base_model = models.densenet121(weights='IMAGENET1K_V1')
        model_name = "DenseNet121"
    else:  # default
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        model_name = "ResNet18"
    
    # Congelar capas del modelo base
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Reemplazar la capa final según el tipo de modelo
    if 'resnet' in experiment_variant or experiment_variant == 'default':
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256), # Simplified architecture
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif experiment_variant == 'mobilenet':
        num_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),  # Simplified architecture
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif experiment_variant == 'efficientnet':
        num_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),  # Simplified architecture
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif experiment_variant == 'densenet':
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    return base_model, model_name

def prepare_data_loaders(data_path, batch_size=32, augmentation_level='medium'):
    """Preparar los cargadores de datos con aumentación de imágenes"""
    # Definir transformaciones según el nivel de aumentación
    if augmentation_level == 'low':
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'high':
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:  # medium (default)
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    # Transformaciones para validación (iguales para todos los niveles)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Crear datasets a partir de las imágenes
    try:
        # Si el dataset ya está dividido en carpetas train/val
        train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), val_transforms)
        
        class_names = list(train_dataset.class_to_idx.keys())
        print(f"Found {len(class_names)} classes in train folder: {class_names}")
    except FileNotFoundError:
        # Si el dataset no está dividido, usamos random_split
        print("Train/Val folders not found. Using the entire dataset and splitting it.")
        
        # Create a dataset from the main folder
        full_dataset = datasets.ImageFolder(data_path, train_transforms)
        
        # Get class names from the dataset
        class_to_idx = full_dataset.class_to_idx
        class_names = list(class_to_idx.keys())
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Apply different transforms to validation set
        # Apply different transforms to validation set
        val_dataset.dataset.transform = val_transforms
    
    
    # Crear data loaders ajustando num_workers según el dispositivo
    # Para MPS, es mejor usar menos workers
    num_workers = 0 if device.type == 'mps' else 2  # Menos workers para MPS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names

def train_model(model, train_loader, val_loader, learning_rate=0.001, epochs=15, 
                patience=3, experiment_name="default", accuracy_threshold=0.82):
    """Entrenar el modelo con parada temprana y seguimiento de métricas"""
    # Inicializar experimento en MLflow
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Registrar parámetros en MLflow
        mlflow.log_param("model_name", experiment_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("patience", patience)
        # Usar el dispositivo global ya definido (CPU, CUDA o MPS)
        print(f"Entrenando en dispositivo: {device}")
        # Usar el dispositivo global ya definido (CPU, CUDA o MPS)
        print(f"Entrenando en dispositivo: {device}")
        model = model.to(device)
        # Función de pérdida y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # Bucle de entrenamiento
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_model_weights = None
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Fase de entrenamiento - Propagación hacia adelante y ajuste de pesos
            model.train()
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Reiniciar gradientes
                optimizer.zero_grad()
                
                # Propagación hacia adelante
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Retropropagación y optimización
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = train_correct / train_total
            
            
            # Fase de validación - Evaluación del modelo con datos no vistos
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_correct / val_total
            
            # Registrar métricas en la iteración actual
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_acc", epoch_val_acc, step=epoch)
            
            # Update learning rate
            scheduler.step(epoch_val_loss)
            
            # Guardar historial ANTES de verificar early stopping
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            
            print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            print("-" * 60)
            
            # Verificación para early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = model.state_dict().copy()
                model_filename = f"{MODEL_PATH.replace('.pth', '')}_{experiment_name}.pth"
                torch.save(model.state_dict(), model_filename)
                best_epoch = epoch
                patience_counter = 0
                print(f"Saved best model with validation loss: {best_val_loss:.4f}, accuracy: {epoch_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    break
                
            # Early stopping si alcanzamos el umbral de precisión deseado
            if epoch_val_acc >= accuracy_threshold:
                print(f"Reached accuracy threshold of {accuracy_threshold:.2%}! Stopping training.")
                break
        # Registrar precisión final
        final_accuracy = history['val_acc'][-1]
        mlflow.log_metric("final_accuracy", final_accuracy)
        
        # Guardar el mejor modelo
        mlflow.pytorch.log_model(model, "model")
        
        # Generar y guardar gráficas
        fig_path = plot_training_history(history, experiment_name)
        mlflow.log_artifact(fig_path)
        
        # Verificar si se alcanzó el objetivo de precisión
        if final_accuracy >= 0.8:
            print(f"✅ Model reached target accuracy of {final_accuracy:.2%} (target: 80%)")
            if final_accuracy >= 0.9:
                print(f"🎉 Model exceeded optimal accuracy of {final_accuracy:.2%} (optimal: 90%)")
        else:
            print(f"❌ Model did not reach target accuracy: {final_accuracy:.2%} (target: 80%)")

        # Check if we hit the early accuracy threshold during training
        if max(history['val_acc']) >= accuracy_threshold:
            print(f"✓ Training stopped early with validation accuracy of {max(history['val_acc']):.2%}")
        
        # Load best model weights
        model.load_state_dict(best_model_weights)
        
        return model, history, final_accuracy

def plot_training_history(history, experiment_name):
    """Generar gráficas del proceso de entrenamiento"""
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de precisión
    ax1.plot(history['train_acc'])
    ax1.plot(history['val_acc'])
    ax1.set_title('Precisión del Modelo')
    ax1.set_ylabel('Precisión')
    ax1.set_xlabel('Época')
    ax1.legend(['Entrenamiento', 'Validación'], loc='upper left')
    
    # Gráfica de pérdida
    ax2.plot(history['train_loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Pérdida del Modelo')
    ax2.set_ylabel('Pérdida')
    ax2.set_xlabel('Época')
    ax2.legend(['Entrenamiento', 'Validación'], loc='upper left')
    
    plt.tight_layout()
    fig_path = f"training_history_{experiment_name}.png"
    plt.savefig(fig_path)
    plt.close()
    
    return fig_path

def save_model_for_inference(model, class_names, experiment_name, final_accuracy):
    """Guardar el modelo para su uso en inferencia si cumple el umbral de precisión"""
    # Save model
    if final_accuracy >= 0.8:
        # Move model to CPU for saving (ensures compatibility)
        model = model.to('cpu')
        
        # Save the model in TorchScript format for deployment
        model.eval()
        example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    
        traced_script_module = torch.jit.trace(model, example)
        
        # Save model with experiment name
        script_module_path = f"model_scripted_{experiment_name}.pt"
        traced_script_module.save(script_module_path)
        
        # Move back to original device
        model = model.to(device)
        return script_module_path
    else:
        print("Model accuracy below threshold, not saving for inference")
        return None

def run_experiment(experiment_variant, learning_rate, augmentation_level, patience):
    """Ejecutar un experimento con los parámetros especificados"""
    try:
        # Set up experiment tracking
        experiment_name = f"{experiment_variant}_lr{learning_rate}_aug{augmentation_level}"
        print(f"\n{'='*20} Running experiment: {experiment_name} {'='*20}\n")
        print(f"\n{'='*20} Ejecutando experimento: {experiment_name} {'='*20}\n")
        # Prepare data loaders
        train_loader, val_loader, class_names = prepare_data_loaders(
            DATASET_PATH, BATCH_SIZE, augmentation_level=augmentation_level
        )
        
        # Create model
        model, model_name = create_model(len(class_names), experiment_variant)
        
        # Train model
        trained_model, history, final_accuracy = train_model(
            model, train_loader, val_loader, 
            learning_rate=learning_rate,
            epochs=EPOCHS,
            patience=patience,
            experiment_name=experiment_name,
            accuracy_threshold=0.82  # Lower threshold but still above 80% requirement
        )
        
        # Save model
        # Guardar modelo
            script_path = save_model_for_inference(trained_model, class_names, experiment_name, final_accuracy)
            if script_path:
                print(f"Saved high-accuracy model: {script_path}")
        print(f"\n{'='*20} Experimento {experiment_name} completado {'='*20}\n")
        return final_accuracy
        
    except Exception as e:
        print(f"Error in experiment {experiment_variant}: {str(e)}")
        return 0.0

def main():
    """Función principal para entrenar y evaluar los modelos"""
    print("Iniciando entrenamiento del modelo...")
    
    # Mostrar información del dispositivo
    print(f"Versión de PyTorch: {torch.__version__}")
    print(f"Dispositivo: {device}")
    if device.type == 'mps':
        print("Usando aceleración GPU de Apple Silicon vía MPS")
    elif device.type == 'cuda':
        print(f"Usando GPU NVIDIA: {torch.cuda.get_device_name(0)}")
    else:
        print("Usando solo CPU - el entrenamiento será más lento")
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("animal_classification")
    
    # Define experiments
    experiments = [
        # Format: (model_variant, learning_rate, augmentation_level, patience)
        ("resnet18", 0.001, "medium", 2),  # Reduced patience to 2
        ("resnet34", 0.001, "medium", 2),
        ("mobilenet", 0.001, "medium", 2),
        ("resnet18", 0.0001, "high", 2),
        ("resnet18", 0.003, "low", 2),
        ("efficientnet", 0.001, "high", 2),
    ]
    
    # Run experiments
    results = []
    for i, (model_variant, lr, aug_level, patience) in enumerate(experiments):
        print(f"\nEjecutando experimento {i+1}/{len(experiments)}")
        accuracy = run_experiment(model_variant, lr, aug_level, patience)
        results.append({
            "experiment": i+1,
            "model": model_variant,
            "learning_rate": lr,
            "augmentation": aug_level,
            "patience": patience,
            "accuracy": accuracy
        })
    
    # Print summary
    print("\n===== RESULTADOS DE EXPERIMENTOS =====")
    for res in results:
        print(f"Experimento {res['experiment']}: {res['model']} - Precisión: {res['accuracy']:.4f}")
    # Encontrar el mejor modelo
    best_experiment = max(results, key=lambda x: x['accuracy'])
    print(f"\nMejor modelo: Experimento {best_experiment['experiment']}")
    print(f"- Modelo: {best_experiment['model']}")
    print(f"- Tasa de aprendizaje: {best_experiment['learning_rate']}")
    print(f"- Aumentación: {best_experiment['augmentation']}")
    print(f"- Precisión: {best_experiment['accuracy']:.4f}")
    print(f"- Dispositivo: {device}")
    
    if best_experiment['accuracy'] >= 0.8:
        print("✅ Precisión objetivo del 80% alcanzada!")
        if best_experiment['accuracy'] >= 0.9:
            print("🎉 Precisión óptima del 90% superada!")
    
    print("\n¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main()
    