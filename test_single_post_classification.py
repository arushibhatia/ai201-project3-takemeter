import os
import json
from typing import List, Literal
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# Load local .env credentials
load_dotenv()

# Initialize Groq client
client = Groq()
MODEL_NAME = "llama-3.3-70b-versatile"

# 1. Structured Validation Contracts
class PostClassification(BaseModel):
    id: str = Field(description="The unique Reddit submission ID.")
    title: str = Field(description="The title of the post.")
    label: Literal["Updates", "Fan Creations", "Poll", "Opinion"] = Field(
        description="The assigned category based on strict criteria rules."
    )
    rationale: str = Field(description="A brief sentence explaining why this label applies based on the edge-case rules.")

class BatchClassificationResponse(BaseModel):
    classifications: List[PostClassification]

# 2. Strict Edge-Case System Prompt
SYSTEM_PROMPT = """
You are an expert data classification bot specialized in text categorization for a music fan community (r/TaylorSwift).
Your task is to classify a batch of Reddit posts into exactly one of four categories: "Updates", "Fan Creations", "Poll", or "Opinion".

CRITERIA DEFINITIONS:
1. Updates: Commercial/chart milestones, official merchandise drops, brand announcements, mainstream media coverage, official studio/tour footage, or organized real-world fan events (e.g., listening parties).
2. Fan Creations: User-generated digital/physical art, covers, instrumentals, tattoo mockups, custom font collections, and detailed fan-made concepts (e.g., hypothetical tracklists, custom album art, "what-if" visual concepts).
3. Poll: Open-ended community discussion questions designed to prompt varied personal replies, favorites, rankings, or hypothetical scenarios from general users.
4. Opinion: Long-form reviews, deep-dive lyrical explanations, structural analyses, essays on writing habits, or thematic arguments.

## 3. Hard Edge Cases

### Opinionated Questions vs. Polls
The most frequent edge case occurs when a user writes a multi-paragraph personal analysis (Opinion) but ends the post title or body text with an open-ended conversational question (e.g., "I think Album X is her undisputed masterpiece. What do you guys think?"). Both Opinion and Poll contain subjective thoughts, but their interactive intent differs.

To handle this consistently during annotation, we will use the following to help us decide:
* If the post text explicitly forces a structural choice, ranking, or data-driven metric ("Pick one,", "Rank these 5 songs,", "Vote on this bracket"), assign the label: Poll
* If the question at the end is more of an invitation for casual conversation or open validation ("Does anyone else agree?", "What are your thoughts?"), the primary volume of text remains a personal monologue. Assign the label: Opinion

### News-Driven Opinion
This is an example that would generally start with a real-world event (which would sound like `Update`), and then shifts to focus on an analysis of this event (which would sound like `Opinion`). For example, "Pitchfork gave the album a 6.5. Personally, I think the reviewer completely missed the emotional depth..."

To handle this, we will use the following to help us decide: if the objective news item is used purely as a catalyst for a personal rant or lyrical critique, label it **`Opinion`**. It should only be labeled an **`Updates`** post if the primary utility of the text is to share raw data, metrics, or official links with minimal user commentary.

---

ADDITIONAL DISAMBIGUATION RULES:
- Media & Covers: An official article or video about an artist is 'Updates'. A user-submitted drum, vocal, or instrument cover of a song is 'Fan Creations'.
- Fan Concepts vs. General Crowd Questions: If a post proposes a highly specific, creative alternate structure (e.g., "What if TTPD was a triple album?" or "My concept for Debut TV"), classify as 'Fan Creations'.

You must return a valid JSON object matching the requested schema. Do not include any markdown formatting code fences or conversational filler outside the raw JSON payload.
"""

def test_single_call():
    # Test a tricky multi-paragraph style title (Case A from edge cases)
    test_post = [{
        "id": "test_edge_case_01",
        "title": "I think The Life of a Showgirl is her most cohesive sonic era yet. What are your thoughts?"
    }]
    
    print(f"Testing Groq pipeline with 1 complex post entry...")
    print(f"Target Input: {test_post[0]['title']}\n")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(test_post, indent=2)}
            ],
            response_format={"type": "json_object", "schema": BatchClassificationResponse.model_json_schema()},
            temperature=0.1
        )
        
        raw_output = completion.choices[0].message.content
        print("--- [RAW API RESPONSE STREAM] ---")
        print(raw_output)
        print("---------------------------------\n")
        
        response_data = json.loads(raw_output)
        
        # --- DEFENSIVE BULLETPROOF PARSING ---
        if isinstance(response_data, list):
            classifications = response_data
        elif "classifications" not in response_data and ("id" in response_data or "ID" in response_data):
            classifications = [response_data]
        else:
            classifications = response_data.get("classifications", [])
        
        if not classifications:
            print("❌ Failure: JSON returned successfully but could not determine structure schema array.")
            return

        item = classifications[0]
        post_id = item.get("id") or item.get("ID")
        label_val = (
            item.get("label") or 
            item.get("Label") or 
            item.get("classification") or 
            item.get("Classification") or 
            ""
        )
        rationale_val = item.get("rationale") or item.get("Rationale") or item.get("reason") or "N/A"
        
        print("--- [PARSED PYTHON DATA OBJECT] ---")
        print(f"Extracted ID: {post_id}")
        print(f"Assigned Label: {label_val}")
        print(f"Model Rationale: {rationale_val}")
        print("-----------------------------------")
        
        if post_id and label_val:
            print("\n✅ Smoke test completely passed! Key structural pairs mapped cleanly.")
        else:
            print("\n❌ Failure: Core structural keys could not be extracted.")

    except Exception as e:
        print(f"\n❌ Pipeline Crash Error: {e}")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Setup Error: Could not find GROQ_API_KEY inside your .env file.")
    else:
        test_single_call()