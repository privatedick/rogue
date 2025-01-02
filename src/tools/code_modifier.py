"""Module for applying AI-generated code changes."""

import os
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from rich.console import Console
from dotenv import load_dotenv
import google.generativeai as genai

# Ladda miljövariabler från .env-filen
load_dotenv()

class CodeModifier:
    """Manages AI-assisted code modifications."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the code modifier.
        
        Args:
            config: AI configuration settings (optional)
        """
        self.config = config or {}
        self.console = Console()
        self._setup_ai()
    
    def _setup_ai(self) -> None:
        """Configure AI with settings."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 64),
            max_output_tokens=self.config.get("max_tokens", 8192),
        )
        
        self.model = genai.GenerativeModel(
            model_name=self.config.get("model", "gemini-2.0-flash-thinking-exp-1219"),
            generation_config=generation_config
        )

    async def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using AI for testing purposes.
        
        Args:
            prompt: The prompt to send to AI
            
        Returns:
            Generated text or None if generation fails
        """
        try:
            response = await self.model.generate_content_async(prompt)
            
            # Check if response was blocked
            if not response.candidates:
                if hasattr(response, 'prompt_feedback'):
                    logging.error(f"Prompt was blocked: {response.prompt_feedback}")
                return None
                
            return response.text if response else None
            
        except Exception as e:
            # Log the specific error type for debugging
            logging.error(f"Text generation failed. Error type: {type(e).__name__}. Error: {str(e)}")
            return None

    async def modify_file(self, file_path: Path, instructions: str) -> bool:
        """Generate and apply modifications to a file.
        
        Args:
            file_path: Path to the file to modify
            instructions: User instructions for modifications
            
        Returns:
            True if modifications were successful
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            original_content = file_path.read_text('utf-8')
            backup_path = None
            
            prompt = f"""Please modify the following code according to these instructions:
            
            Instructions:
            {instructions}
            
            Current code:
            {original_content}
            
            Please provide the complete modified code:"""
            
            # Create backup before modification
            backup_path = self.create_backup(file_path)
            
            try:
                modified_content = await self.generate_text(prompt)
                if not modified_content:
                    raise ValueError("No content generated")
                
                # Write modified content
                file_path.write_text(modified_content, encoding='utf-8')
                return True
                
            except Exception as e:
                # Restore from backup if anything goes wrong
                if backup_path and backup_path.exists():
                    backup_path.rename(file_path)
                raise e
            
        except Exception as e:
            logging.error(f"Error modifying file {file_path}: {e}")
            return False

    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.with_suffix(f'.{timestamp}.bak')
        file_path.write_bytes(file_path.read_bytes())
        return backup_path
