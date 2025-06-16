# utils/llm_interface.py
import google.generativeai as genai
from typing import Optional, Dict, Any
import time

try:
    from config import APP_CONFIG
except ImportError:
    print("Warning: Could not import APP_CONFIG from config.py. GEMINI_API_KEY must be set via other means if not testing.")
    # For standalone testing, you might set a dummy key or load .env directly here
    import os
    from dotenv import load_dotenv
    load_dotenv()
    class DummyConfig:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    APP_CONFIG = DummyConfig()


if APP_CONFIG.GEMINI_API_KEY:
    try:
        genai.configure(api_key=APP_CONFIG.GEMINI_API_KEY)
    except Exception as e:
        print(f"Error configuring Google Generative AI: {e}. Ensure GEMINI_API_KEY is valid.")
else:
    print("Warning: GEMINI_API_KEY not found in APP_CONFIG. LLM functionality will be disabled.")

DEFAULT_MODEL_NAME = "gemma-3-27b-it" # User should verify this or change if using a custom Gemma endpoint

def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: int = 5, # seconds
    generation_config_override: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generates text using the configured Gemini model.

    Args:
        prompt: The main prompt/query for the LLM.
        model_name: The specific model to use (e.g., "gemini-pro").
        system_instruction: An optional system-level instruction to guide the model's behavior.
                            (Note: `google-generativeai` handles system instructions differently
                             depending on the model version. For newer models, it's part of the content.)
        temperature: Controls randomness (0.0 to 1.0). Lower is more deterministic.
        max_retries: Number of times to retry on failure.
        retry_delay: Delay between retries in seconds.
        generation_config_override: Advanced generation parameters to override defaults.

    Returns:
        The generated text as a string, or None if an error occurs after retries.
    """
    if not APP_CONFIG.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not configured. Cannot generate text.")
        return None

    try:
        model = genai.GenerativeModel(
            model_name,
            # system_instruction is handled differently in genai vs. e.g. OpenAI's API
            # For Gemini, system instructions are often prepended to the user prompt
            # or passed via specific `Content` object structures.
            # The `system_instruction` parameter in `GenerativeModel` is for some specific models.
        )
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        return None

    # Construct contents. System instructions can be passed as a specific part if model supports.
    # For many Gemini models, you'd prepend system instructions to the user prompt or use a multi-turn chat structure.
    # Simplified approach: prepend if provided.
    full_prompt_parts = []
    if system_instruction:
        # The genai library handles system_instruction if the model supports it directly.
        # Some models prefer it as the first message in a chat history.
        # For a simple generation, we might need to adjust how it's passed.
        # Let's assume model.generate_content can take a list of parts, or we prepend.
        # For "gemini-pro" via `generate_content`, it's often just part of the prompt string.
        # If a `system_instruction` argument is directly supported by `GenerativeModel` init for Gemma,
        # that's preferred. Let's assume for now we prepend.
        # A more robust way for models like Gemini 1.5 Pro might be:
        # model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        # However, to keep it general for "gemini-pro" too:
        prompt = f"{system_instruction}\n\nUser Prompt: {prompt}"


    generation_config = genai.types.GenerationConfig(
        temperature=temperature
        # Add other parameters like top_p, top_k, max_output_tokens if needed
    )
    if generation_config_override:
        generation_config.update(generation_config_override)


    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt, # Or `full_prompt_parts` if structured differently
                generation_config=generation_config
            )
            # Handle potential safety blocks or empty responses
            if response.parts:
                return response.text
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Warning: Prompt blocked due to {response.prompt_feedback.block_reason}. Content: {prompt[:200]}...")
                return f"Error: Content generation blocked due to {response.prompt_feedback.block_reason}."
            else:
                print(f"Warning: Received empty response from LLM for prompt: {prompt[:200]}...")
                # Consider if this should be a retryable error or a specific return
                return None # Or an empty string, or a specific error message

        except Exception as e:
            print(f"Error generating text (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Failed to generate text.")
                return None
    return None

if __name__ == '__main__':
    # Ensure you have a .env file with GEMINI_API_KEY in the project root for this test
    print("Testing LLM Interface...")
    if not APP_CONFIG.GEMINI_API_KEY:
        print("Skipping LLM test: GEMINI_API_KEY not found.")
    else:
        print(f"Using Gemini API Key: ...{APP_CONFIG.GEMINI_API_KEY[-4:]}") # Masked key
        
        # Test with a system prompt from file
        from .prompt_loader import get_prompt_template
        system_persona_data = get_prompt_template("default_system_message", "system_prompts.yaml")
        system_instruction_text = system_persona_data.get("default_system_message") if system_persona_data else None

        test_prompt = "What is the capital of France?"
        print(f"\nSending prompt: '{test_prompt}' with system instruction: '{system_instruction_text}'")
        
        response_text = generate_text(test_prompt, system_instruction=system_instruction_text)
        if response_text:
            print(f"LLM Response:\n{response_text}")
        else:
            print("Failed to get response from LLM.")

        print("\nTesting summarization prompt...")
        from .prompt_loader import format_prompt
        long_text = "Python is a versatile and widely-used programming language known for its readability and extensive libraries. It supports multiple programming paradigms, including object-oriented, imperative, and functional programming. Python's large standard library and third-party packages make it suitable for web development, data science, artificial intelligence, scientific computing, and more. Its simple syntax allows for rapid development and prototyping."
        
        summarization_prompt = format_prompt(
            "summarize_text_generic",
            "task_specific_prompts.yaml",
            text_content=long_text
        )
        if summarization_prompt:
            print(f"\nSending summarization prompt...")
            summary_response = generate_text(summarization_prompt, temperature=0.5)
            if summary_response:
                print(f"LLM Summary Response:\n{summary_response}")
            else:
                print("Failed to get summary from LLM.")
        else:
            print("Failed to format summarization prompt.")