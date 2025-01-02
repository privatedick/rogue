"""Module for applying AI-generated code changes to files.

This module can run both as a standalone script and as part of the larger project.
It provides functionality for AI-assisted code modifications with fallback options
for when the main project infrastructure is not available.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime

try:
    # Try to import project dependencies
    from rich.console import Console
    from rich.prompt import Confirm
    from dotenv import load_dotenv
    from google import generativeai as genai
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: Running in minimal mode without rich UI and other dependencies")

# Setup basic logging first - will be enhanced if project infrastructure is available
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModifierConfig:
    """Configuration for the code modifier.
    
    This provides a simplified version of the project's configuration
    for standalone usage.
    """
    model_name: str = "gemini-2.0-flash-thinking-exp-1219"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 8192

class StandaloneAIManager:
    """Simplified AI manager for standalone operation."""
    
    def __init__(self, config: ModifierConfig):
        """Initialize the standalone AI manager.
        
        Args:
            config: Configuration settings
        """
        self.config = config
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Configure the AI model with basic settings."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_tokens,
            }
        )
    
    async def generate_code(self, prompt: str) -> Optional[str]:
        """Generate code using the AI model.
        
        Args:
            prompt: Code generation prompt
            
        Returns:
            Generated code or None if generation fails
        """
        try:
            response = await self.model.generate_content(prompt)
            return response.text if response else None
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None

class CodeModifier:
    """Manages AI-assisted code modifications."""
    
    def __init__(self, ai_manager: Any = None, config: Optional[ModifierConfig] = None):
        """Initialize the code modifier.
        
        Args:
            ai_manager: AI manager instance (optional)
            config: Configuration for standalone mode (optional)
        """
        self.project_mode = ai_manager is not None
        
        if self.project_mode:
            self.ai_manager = ai_manager
        else:
            self.ai_manager = StandaloneAIManager(config or ModifierConfig())
        
        self.console = Console() if HAS_DEPENDENCIES else None
        
    def _print(self, message: str, style: Optional[str] = None) -> None:
        """Print messages with rich formatting if available.
        
        Args:
            message: Message to print
            style: Rich style to apply (ignored in minimal mode)
        """
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def _confirm(self, question: str) -> bool:
        """Get user confirmation with rich prompt if available.
        
        Args:
            question: Question to ask
            
        Returns:
            User's response
        """
        if HAS_DEPENDENCIES:
            return Confirm.ask(question)
        else:
            response = input(f"{question} (y/N): ").lower()
            return response in ('y', 'yes')

    async def modify_file(self, file_path: Union[str, Path], instructions: str) -> Optional[str]:
        """Generate and apply modifications to a file.
        
        Args:
            file_path: Path to the file to modify
            instructions: User instructions for modifications
            
        Returns:
            Modified content if successful, None otherwise
        """
        try:
            file_path = Path(file_path)
            original_content = file_path.read_text(encoding='utf-8')
            
            prompt = f"""Please modify the following code according to the instructions.
            Keep the file's original structure and documentation style.
            
            File: {file_path.name}
            
            Original content:
            {original_content}
            
            Instructions:
            {instructions}
            
            Please provide the complete modified code:"""
            
            modified_content = await self.ai_manager.generate_code(prompt)
            
            if modified_content:
                return modified_content
            
            logger.error(f"Failed to generate modifications for {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error modifying file {file_path}: {e}")
            return None

    async def apply_changes(self, file_path: Union[str, Path], modified_content: str) -> bool:
        """Apply generated changes to the file after confirmation.
        
        Args:
            file_path: Path to the file to modify
            modified_content: New content to write
            
        Returns:
            True if changes were applied successfully
        """
        try:
            file_path = Path(file_path)
            
            # Backup original file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f".{timestamp}.bak")
            file_path.rename(backup_path)
            
            self._print("\nProposed changes:", style="yellow" if self.console else None)
            self._print(modified_content)
            
            if self._confirm("\nDo you want to apply these changes?"):
                file_path.write_text(modified_content, encoding='utf-8')
                self._print(
                    f"\nSuccessfully modified {file_path}",
                    style="green" if self.console else None
                )
                return True
            
            # Restore from backup if changes are rejected
            backup_path.rename(file_path)
            self._print(
                "\nChanges were not applied", 
                style="yellow" if self.console else None
            )
            return False
            
        except Exception as e:
            logger.error(f"Error applying changes to {file_path}: {e}")
            if 'backup_path' in locals():
                try:
                    backup_path.rename(file_path)
                except Exception:
                    logger.error("Failed to restore from backup")
            return False

async def main():
    """Main function for standalone operation."""
    if not HAS_DEPENDENCIES:
        print("Error: Required dependencies not installed.")
        print("Run: pip install rich python-dotenv google-generativeai")
        return

    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    modifier = CodeModifier()
    
    file_path = input("Enter path to the file to modify: ")
    instructions = input("Enter modification instructions: ")
    
    modified_content = await modifier.modify_file(file_path, instructions)
    if modified_content:
        await modifier.apply_changes(file_path, modified_content)

if __name__ == "__main__":
    asyncio.run(main())
