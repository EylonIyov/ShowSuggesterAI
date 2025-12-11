"""Module for generating creative TV show recommendations using LLM."""

import os
from typing import List, Tuple, Dict
from pathlib import Path
import dotenv
from openai import OpenAI
from google import genai
from google.genai import types

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def generate_creative_shows(
    user_shows: List[str],
    user_show_descriptions: List[str],
    recommended_shows: List[Tuple[str, float]],
) -> Tuple[str, Dict[str, str]]:
    """Generate two creative fictional shows based on user preferences.
    
    Args:
        user_shows: List of show names the user provided.
        user_show_descriptions: List of descriptions for user's favorite shows.
        recommended_shows: List of tuples (show_name, similarity_score) from recommendations.
    
    Returns:
        Tuple of (formatted_output_string, shows_data_dict) where shows_data_dict contains
        show1_name, show1_desc, show2_name, show2_desc.
    """
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please configure your API key in the .env file."
        )
    
    client = OpenAI(api_key=api_key)
    
    # Prepare input data for the prompt
    user_shows_str = ", ".join(user_shows)
    user_descriptions_str = "\n".join(
        [f"- {show}: {desc}" for show, desc in zip(user_shows, user_show_descriptions)]
    )
    
    recommended_shows_str = ", ".join([show[0] for show in recommended_shows])
    
    # Create a detailed prompt for the LLM
    prompt = f"""You are a creative TV show writer. Based on the following information, create TWO completely fictional TV shows:

User's Favorite Shows: {user_shows_str}
Their Descriptions:
{user_descriptions_str}

Recommended Shows to Consider: {recommended_shows_str}

Please generate:
1. Show #1: A fictional show inspired by the user's favorite shows
2. Show #2: A fictional show inspired by the recommended shows

For each show, provide:
- A creative show name (must be a unique, fictional name NOT an existing show)
- A compelling 1-2 sentence description about what the show is about

Format your response EXACTLY as:
SHOW1_NAME: [creative name here]
SHOW1_DESC: [1-2 sentence description]
SHOW2_NAME: [creative name here]
SHOW2_DESC: [1-2 sentence description]

Make sure the descriptions are engaging and capture the essence of what makes those shows appealing."""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": "You are a creative TV show writer who generates unique, fictional show concepts.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # Higher creativity
    )
    
    # Parse the response
    llm_output = response.choices[0].message.content
    shows_data = _parse_llm_response(llm_output)
    
    # Format the final output message
    output_message = (
        f"I have also created just for you two shows which I think you would love.\n"
        f"Show #1 is based on the fact that you loved the input shows that you gave me. "
        f"Its name is {shows_data['show1_name']} and it is about {shows_data['show1_desc']}.\n"
        f"Show #2 is based on the shows that I recommended for you. Its name is "
        f"{shows_data['show2_name']} and it is about {shows_data['show2_desc']}.\n"
        f"Here are also the 2 tv show ads. Hope you like them!"
    )
    
    return output_message, shows_data


def _parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response to extract show names and descriptions.
    
    Args:
        response_text: Raw text response from the LLM.
    
    Returns:
        Dictionary with keys: show1_name, show1_desc, show2_name, show2_desc
    """
    result = {
        "show1_name": "Creative Show #1",
        "show1_desc": "An amazing new series",
        "show2_name": "Creative Show #2",
        "show2_desc": "Another fantastic show",
    }
    
    lines = response_text.strip().split("\n")
    
    for line in lines:
        if "SHOW1_NAME:" in line:
            result["show1_name"] = line.split("SHOW1_NAME:", 1)[1].strip()
        elif "SHOW1_DESC:" in line:
            result["show1_desc"] = line.split("SHOW1_DESC:", 1)[1].strip()
        elif "SHOW2_NAME:" in line:
            result["show2_name"] = line.split("SHOW2_NAME:", 1)[1].strip()
        elif "SHOW2_DESC:" in line:
            result["show2_desc"] = line.split("SHOW2_DESC:", 1)[1].strip()
    
    return result


def generate_show_images(shows_data: Dict[str, str]) -> Dict[str, str]:
    """Generate images for the creative shows using Google Gemini API.
    
    Args:
        shows_data: Dictionary with keys show1_name, show1_desc, show2_name, show2_desc
    
    Returns:
        Dictionary with image file paths: {show1_image_path, show2_image_path}
    """
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create output directory for images
    output_dir = Path(__file__).parent.parent / "generated_images"
    output_dir.mkdir(exist_ok=True)
    
    result = {}
    
    # Generate image for Show #1
    print("Generating image for Show #1...")
    show1_prompt = f"Create a professional promotional movie poster for the TV show '{shows_data['show1_name']}':\n{shows_data['show1_desc']}"
    show1_image_path = _generate_and_save_image_gemini(client, show1_prompt, output_dir, shows_data['show1_name'])
    result["show1_image_path"] = show1_image_path
    
    # Generate image for Show #2
    print("Generating image for Show #2...")
    show2_prompt = f"Create a professional promotional movie poster for the TV show '{shows_data['show2_name']}':\n{shows_data['show2_desc']}"
    show2_image_path = _generate_and_save_image_gemini(client, show2_prompt, output_dir, shows_data['show2_name'])
    result["show2_image_path"] = show2_image_path
    
    return result


def _generate_and_save_image_gemini(client: genai.Client, prompt: str, output_dir: Path, show_name: str) -> str:
    """Generate an image using Google Gemini API and save it to disk.
    
    Args:
        client: Gemini client instance.
        prompt: Text prompt for image generation.
        output_dir: Directory to save the image in.
        show_name: Name of the show (used for filename).
    
    Returns:
        Path to the saved image file.
    
    Raises:
        RuntimeError: If image generation or saving fails.
    """
    try:
        # Call Gemini API to generate image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )
        
        # Extract and save the image
        for part in response.parts:
            if part.inline_data is not None:
                # Get the image and save it
                image = part.as_image()
                
                # Create a safe filename from the show name
                safe_name = "".join(c for c in show_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                safe_name = safe_name.replace(" ", "_")
                filename = f"{safe_name}.png"
                file_path = output_dir / filename
                
                # Save the image
                image.save(str(file_path))
                print(f"Image saved: {file_path}")
                return str(file_path)
        
        raise RuntimeError("No image data found in Gemini API response")
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate image with Gemini API: {str(e)}")





def open_generated_images(image_paths: Dict[str, str]) -> None:
    """Open the generated images in the default image viewer.
    
    Args:
        image_paths: Dictionary with image file paths (show1_image_path, show2_image_path).
    """
    import subprocess
    import platform
    
    for key, image_path in image_paths.items():
        if image_path and Path(image_path).exists():
            try:
                abs_path = str(Path(image_path).resolve())
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    subprocess.run(["open", abs_path], check=True)
                elif system == "Windows":
                    # Use subprocess with shell=True for Windows
                    subprocess.run(["cmd", "/c", "start", "", abs_path], check=True)
                elif system == "Linux":
                    subprocess.run(["xdg-open", abs_path], check=True)
                else:
                    print(f"Unsupported platform: {system}")
                    continue
                    
                print(f"Opened {key}: {abs_path}")
            except Exception as e:
                print(f"Could not open {image_path}: {str(e)}")
        else:
            print(f"Image file not found: {image_path}")

