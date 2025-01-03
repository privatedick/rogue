from google import generativeai as genai
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import os
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIManager:
    """Manages AI model interactions."""

    MODEL_CONFIGS = {
        "gemini-2.0-flash-exp": {
            "max_tokens": 1000000,
            "best_for": ["large_files", "quick_edits"],
        },
        "gemini-2.0-flash-thinking-exp-1219": {
            "max_tokens": 32000,
            "best_for": ["complex_analysis", "code_generation"],
        },
    }

    def __init__(self, config: Dict[str, Any], preferred_model: Optional[str] = None):
        """Initialize AI manager.

        Args:
            config: AI configuration settings
            preferred_model: Specific model to use; selects best model if None.
        """
        self.config = config
        self.preferred_model = preferred_model
        self._setup_model()

    def _setup_model(self) -> None:
        """Configure AI model with settings."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = self.preferred_model or self.select_best_model(task_type="default")
        self._validate_model(model_name)
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.config,
        )
        logging.info(f"Model '{self.model_name}' initialized with config: {self.config}")

    def select_best_model(self, task_type: str) -> str:
        """Select the best model based on the task type."""
        if task_type in ["large_files", "quick_edits"]:
            return "gemini-2.0-flash-exp"
        elif task_type in ["complex_analysis", "code_generation"]:
            return "gemini-2.0-flash-thinking-exp-1219"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _validate_model(self, model_name: str) -> None:
        """Validate the chosen model."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model name: {model_name}")
        logging.info(f"Using model '{model_name}' for current task.")

    def validate_token_count(self, content: str, model: str) -> bool:
        """Ensure the content fits within the model's token limit."""
        max_tokens = self.MODEL_CONFIGS[model]["max_tokens"]
        token_count = len(content.split())  # Approximate token count
        if token_count > max_tokens:
            logging.error(
                f"Content exceeds token limit for model '{model}': {token_count}/{max_tokens} tokens."
            )
            return False
        return True

    def handle_thinking_response(self, response) -> Tuple[str, Optional[str]]:
        """Handle thinking model response, separating content and thoughts."""
        thoughts = ""
        edited_content = ""

        for part in response.candidates[0].content.parts:
            if part.thought:
                thoughts += part.text + "\n"
            else:
                edited_content += part.text

        return edited_content, thoughts if thoughts else None

    async def generate_code(self, prompt: str, file_path: Path, task_type: str = "default") -> bool:
        """Generate code using AI model and write to a file.

        Args:
            prompt: Code generation prompt
            file_path: Path to the file where code will be written
            task_type: Type of task to determine the model to use

        Returns:
            True if code generation and file writing are successful, False otherwise
        """
        self.model_name = self.select_best_model(task_type)
        self._setup_model()

        if not self.validate_token_count(prompt, self.model_name):
            return False

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        branch_name = f"ai_mod_{timestamp}"

        if not self._create_git_branch(branch_name):
            return False

        try:
            response = await self.model.generate_content(prompt)
            if response and response.text:
                content, thoughts = self.handle_thinking_response(response)
                with open(file_path, "w") as f:
                    f.write(content)
                if thoughts:
                    with open(f"{file_path.stem}_thoughts.txt", "w") as tf:
                        tf.write(thoughts)
                logging.info(f"Successfully generated code and wrote to {file_path}")
                return True
            else:
                logging.error("Code generation failed: No response or empty response received.")
                self._handle_modification_failure(branch_name)
                return False
        except Exception as e:
            logging.error(f"Code generation or file writing failed: {e}")
            self._handle_modification_failure(branch_name)
            return False

    def _run_git_command(self, command: list) -> bool:
        """Execute a git command and log the output."""
        process = subprocess.run(command, capture_output=True, text=True)
        logging.info(f"Executing git command: {' '.join(command)}")
        if process.returncode == 0:
            logging.info(f"Git command successful.\nStdout: {process.stdout.strip()}")
            return True
        else:
            logging.error(f"Git command failed.\nStderr: {process.stderr.strip()}")
            return False

    def _create_git_branch(self, branch_name: str) -> bool:
        """Create a new git branch."""
        command = ["git", "checkout", "-b", branch_name]
        return self._run_git_command(command)

    def _switch_to_main_branch(self) -> bool:
        """Switch back to the main branch."""
        command = ["git", "checkout", "main"]
        return self._run_git_command(command)

    def _delete_git_branch(self, branch_name: str) -> bool:
        """Delete the specified git branch."""
        command = ["git", "branch", "-D", branch_name]
        return self._run_git_command(command)

    def _handle_modification_failure(self, branch_name: str) -> None:
        """Handle the failure of code modification."""
        logging.info("Modification failed. Switching back to main branch and deleting temporary branch.")
        if self._switch_to_main_branch():
            if self._delete_git_branch(branch_name):
                logging.info(f"Successfully switched back to main and deleted branch '{branch_name}'.")
            else:
                logging.error(f"Failed to delete branch '{branch_name}'.")
        else:
            logging.error("Failed to switch back to the main branch.")
