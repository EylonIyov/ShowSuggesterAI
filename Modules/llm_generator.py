"""Module for generating creative TV show recommendations using LLM."""

import os
from typing import List, Tuple
import dotenv
from openai import OpenAI

dotenv.load_dotenv()


def generate_creative_shows(
    user_shows: List[str],
    user_show_descriptions: List[str],
    recommended_shows: List[Tuple[str, float]],
) -> str:
    """Generate two creative fictional shows based on user preferences.
    
    Args:
        user_shows: List of show names the user provided.
        user_show_descriptions: List of descriptions for user's favorite shows.
        recommended_shows: List of tuples (show_name, similarity_score) from recommendations.
    
    Returns:
        A formatted string with two creative show recommendations and descriptions.
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
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a creative TV show writer who generates unique, fictional show concepts.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # Higher creativity
        max_tokens=500,
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
    
    return output_message


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
