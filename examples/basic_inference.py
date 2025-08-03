#!/usr/bin/env python3
"""
Basic inference example for Qwen3-Coder-480B-A35B-Instruct

This script demonstrates how to load and use the model for code generation.
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenCodeGenerator:
    """
    A simple wrapper for Qwen3-Coder-480B inference.
    """
    
    def __init__(self, model_path=None, device="auto"):
        """
        Initialize the code generator.
        
        Args:
            model_path (str): Path to the model directory
            device (str): Device to run the model on
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def _get_default_model_path(self):
        """Get the default model path from environment."""
        install_dir = os.environ.get('INSTALL_DIR')
        if install_dir:
            return os.path.join(install_dir, 'models', 'qwen3-coder-480b')
        return None
    
    def load_model(self):
        """Load the model and tokenizer."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"Model path not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
        
        # Load model
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
    
    def generate_code(self, prompt, max_tokens=200, temperature=0.7, top_p=0.9):
        """
        Generate code based on the prompt.
        
        Args:
            prompt (str): The input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            dict: Generated response with metadata
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        inference_time = time.time() - start_time
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):]
        
        # Calculate metrics
        new_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = new_tokens / inference_time if inference_time > 0 else 0
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_response': full_response,
            'inference_time': inference_time,
            'tokens_generated': new_tokens,
            'tokens_per_second': tokens_per_second,
        }

def main():
    """Main function to demonstrate basic usage."""
    print("🚀 Qwen3-Coder-480B Basic Inference Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = QwenCodeGenerator()
    
    try:
        # Load model
        generator.load_model()
        
        # Example prompts
        test_prompts = [
            "Write a Python function to calculate the factorial of a number:",
            "Implement a binary search algorithm in C++:",
            "Create a simple REST API endpoint in JavaScript using Express:",
            "Write a SQL query to find the top 5 customers by total order value:",
            "Implement a merge sort algorithm in Rust:",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}/{len(test_prompts)}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print("-" * 60)
            
            # Generate response
            result = generator.generate_code(prompt, max_tokens=150)
            
            print(f"Generated Code:")
            print(result['generated_text'])
            
            print(f"\nMetrics:")
            print(f"  Inference time: {result['inference_time']:.2f}s")
            print(f"  Tokens generated: {result['tokens_generated']}")
            print(f"  Speed: {result['tokens_per_second']:.1f} tokens/sec")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()