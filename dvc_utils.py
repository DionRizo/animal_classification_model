#!/usr/bin/env python3
"""
Data Version Control (DVC) utility functions for the Animal Classification Model.

This module provides functionality to manage data and model versioning using DVC.
It includes methods for:
- Initializing a DVC repository
- Adding and updating datasets to DVC
- Creating and managing model training pipelines
- Reproducing experiments and comparing results
- Managing model versions

Usage:
    import dvc
    dvc.init_dvc_repo()  # Initialize DVC in your project
    dvc.add_data("data/raw")  # Add raw data to DVC tracking
    dvc.create_pipeline("train_model.dvc", dependencies=["data/processed"], outputs=["models/"])
"""

import os
import subprocess
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dvc_utils')

def init_dvc_repo() -> bool:
    """
    Initialize a DVC repository in the current directory if it doesn't exist.
    
    Returns:
        bool: True if initialization was successful or already initialized, False otherwise
    """
    try:
        if os.path.exists('.dvc'):
            logger.info("DVC repository already initialized.")
            return True
        
        logger.info("Initializing DVC repository...")
        subprocess.run(['dvc', 'init'], check=True)
        logger.info("DVC repository initialized successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize DVC repository: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during DVC initialization: {e}")
        return False

def add_data(data_path: str) -> bool:
    """
    Add data to DVC tracking.
    
    Args:
        data_path (str): Path to the data directory or file
        
    Returns:
        bool: True if data was successfully added to DVC, False otherwise
    """
    try:
        if not os.path.exists(data_path):
            logger.error(f"Data path '{data_path}' does not exist.")
            return False
        
        logger.info(f"Adding '{data_path}' to DVC...")
        subprocess.run(['dvc', 'add', data_path], check=True)
        logger.info(f"Successfully added '{data_path}' to DVC.")
        
        # Commit the .dvc file to git if git is being used
        if os.path.exists('.git'):
            dvc_file = f"{data_path}.dvc"
            subprocess.run(['git', 'add', dvc_file], check=True)
            logger.info(f"Added {dvc_file} to git staging.")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add data to DVC: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when adding data to DVC: {e}")
        return False

def create_pipeline(
    stage_file: str, 
    command: str, 
    dependencies: List[str],
    outputs: List[str],
    params: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    description: Optional[str] = None
) -> bool:
    """
    Create or update a DVC pipeline stage.
    
    Args:
        stage_file (str): Name of the .dvc file for this stage
        command (str): Command to execute for this stage
        dependencies (List[str]): List of dependencies (inputs)
        outputs (List[str]): List of outputs produced by this stage
        params (Optional[List[str]]): List of parameters used by this stage
        metrics (Optional[List[str]]): List of metrics files generated
        description (Optional[str]): Description of the stage
        
    Returns:
        bool: True if the pipeline stage was created successfully, False otherwise
    """
    try:
        cmd = ['dvc', 'stage', 'add', '--name', os.path.splitext(stage_file)[0]]
        
        # Add dependencies
        for dep in dependencies:
            cmd.extend(['--deps', dep])
            
        # Add outputs
        for out in outputs:
            cmd.extend(['--outs', out])
            
        # Add parameters if provided
        if params:
            for param in params:
                cmd.extend(['--params', param])
                
        # Add metrics if provided
        if metrics:
            for metric in metrics:
                cmd.extend(['--metrics', metric])
                
        # Add description if provided
        if description:
            cmd.extend(['--desc', description])
            
        # Add the command to run
        cmd.append(command)
        
        logger.info(f"Creating DVC pipeline stage: {stage_file}")
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully created DVC pipeline stage: {stage_file}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create DVC pipeline stage: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when creating DVC pipeline: {e}")
        return False

def reproduce_pipeline(stage_file: str) -> bool:
    """
    Reproduce a DVC pipeline stage.
    
    Args:
        stage_file (str): Name of the .dvc file to reproduce
        
    Returns:
        bool: True if the pipeline was reproduced successfully, False otherwise
    """
    try:
        logger.info(f"Reproducing DVC pipeline: {stage_file}")
        subprocess.run(['dvc', 'repro', stage_file], check=True)
        logger.info(f"Successfully reproduced DVC pipeline: {stage_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reproduce DVC pipeline: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when reproducing pipeline: {e}")
        return False

def push_data_to_remote() -> bool:
    """
    Push data to the configured DVC remote storage.
    
    Returns:
        bool: True if data was pushed successfully, False otherwise
    """
    try:
        logger.info("Pushing data to remote storage...")
        subprocess.run(['dvc', 'push'], check=True)
        logger.info("Successfully pushed data to remote storage.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to push data to remote: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error when pushing data: {e}")
        return False

def get_metrics(stage_file: str) -> Dict[str, Any]:
    """
    Get metrics for a specific DVC pipeline stage.
    
    Args:
        stage_file (str): Name of the .dvc file
        
    Returns:
        Dict[str, Any]: Dictionary containing metrics values
    """
    try:
        result = subprocess.run(['dvc', 'metrics', 'show', stage_file, '--json'], 
                               check=True, capture_output=True, text=True)
        
        import json
        metrics = json.loads(result.stdout)
        logger.info(f"Retrieved metrics for {stage_file}")
        return metrics
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get metrics: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error when getting metrics: {e}")
        return {}

def setup_animal_classification_pipeline() -> bool:
    """
    Set up the complete animal classification model pipeline.
    This includes data preparation, feature extraction, model training and evaluation.
    
    Returns:
        bool: True if the pipeline was set up successfully, False otherwise
    """
    try:
        # Initialize DVC repository
        if not init_dvc_repo():
            return False
            
        # Create data preparation stage
        create_pipeline(
            stage_file="prepare_data.dvc",
            command="python scripts/prepare_data.py",
            dependencies=["data/raw"],
            outputs=["data/processed"],
            description="Prepare and preprocess raw animal images"
        )
        
        # Create model training stage
        create_pipeline(
            stage_file="train_model.dvc",
            command="python animal_classification_model_training.py",
            dependencies=["data/processed"],
            outputs=["models/animal_classifier.pt"],
            params=["params.yaml:training"],
            metrics=["metrics/training_metrics.json"],
            description="Train the animal classification model"
        )
        
        # Create evaluation stage
        create_pipeline(
            stage_file="evaluate_model.dvc",
            command="python scripts/evaluate_model.py",
            dependencies=["models/animal_classifier.pt", "data/processed/test"],
            outputs=[],
            metrics=["metrics/evaluation_metrics.json"],
            description="Evaluate the animal classification model"
        )
        
        logger.info("Successfully set up animal classification pipeline.")
        return True
    except Exception as e:
        logger.error(f"Failed to set up animal classification pipeline: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("DVC Utility Functions for Animal Classification Model")
    print("Use this module by importing it in your scripts.")

