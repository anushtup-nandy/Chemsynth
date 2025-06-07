# utils/prompt_loader.py
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

PROMPT_DIR = Path(__file__).parent / "prompts"

def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Loads a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return {}

def get_prompt_template(
    prompt_key: str,
    file_name: str = "task_specific_prompts.yaml"
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific prompt template from a YAML file.

    Args:
        prompt_key: The key of the prompt in the YAML file.
        file_name: The name of the YAML file in the 'prompts' directory.

    Returns:
        A dictionary containing the prompt template or None if not found.
    """
    file_path = PROMPT_DIR / file_name
    prompts = load_yaml_file(file_path)
    return prompts.get(prompt_key)

def format_prompt(prompt_key: str, file_name: str = "task_specific_prompts.yaml", **kwargs) -> Optional[str]:
    """
    Loads a prompt template, formats it with provided keyword arguments, and returns the formatted prompt string.

    Args:
        prompt_key: The key of the prompt in the YAML file.
        file_name: The name of the YAML file.
        **kwargs: The values to substitute into the prompt template's input variables.

    Returns:
        The formatted prompt string or None if the template is not found or formatting fails.
    """
    template_data = get_prompt_template(prompt_key, file_name)
    if not template_data or "prompt" not in template_data:
        print(f"Error: Prompt template '{prompt_key}' not found or is invalid in {file_name}.")
        return None

    prompt_template_str = template_data["prompt"]
    expected_vars = template_data.get("input_variables", [])

    try:
        # Basic check if all expected variables are provided, though str.format handles missing ones with KeyError
        for var in expected_vars:
            if var not in kwargs:
                print(f"Warning: Expected variable '{var}' not provided for prompt '{prompt_key}'.")
        
        return prompt_template_str.format(**kwargs)
    except KeyError as e:
        print(f"Error formatting prompt '{prompt_key}': Missing key {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during prompt formatting for '{prompt_key}': {e}")
        return None

if __name__ == '__main__':
    # Example Usage:
    # Ensure your_project_name/config.py exists and is accessible if other modules depend on it
    # For this test, we only need the prompt files.

    # Test loading a system prompt (though typically used directly, not formatted)
    system_persona = get_prompt_template("scientific_analyzer_persona", "system_prompts.yaml")
    if system_persona:
        print(f"System Persona: {system_persona.get('role', system_persona.get('default_system_message'))}\n")

    # Test formatting a task-specific prompt
    summary_prompt_text = format_prompt(
        "summarize_text_generic",
        text_content="This is a sample text that needs to be summarized by the LLM."
    )
    if summary_prompt_text:
        print(f"Formatted Summary Prompt:\n{summary_prompt_text}\n")

    arxiv_abstract = "We present a novel method for..." # A short dummy abstract
    arxiv_summary_prompt = format_prompt(
        "summarize_arxiv_abstract",
        "task_specific_prompts.yaml",
        abstract_text=arxiv_abstract
    )
    if arxiv_summary_prompt:
        print(f"Formatted ArXiv Summary Prompt:\n{arxiv_summary_prompt}\n")
    
    non_existent_prompt = format_prompt("i_do_not_exist", text_content="...")
    if not non_existent_prompt:
        print("Successfully handled non-existent prompt key.\n")

    prompt_missing_var = format_prompt("summarize_text_generic") # Missing 'text_content'
    # The warning should be printed by the function itself.