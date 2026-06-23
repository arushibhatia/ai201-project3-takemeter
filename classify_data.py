import os
import json
import time
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Initialize Groq client
client = Groq()

INPUT_CSV = "taylor_swift_dataset.csv"
MODEL_NAME = "llama-3.3-70b-versatile"
BATCH_SIZE = 10  

# --- TESTING CONTROL FLAG ---
# Set to None when you're ready to let it run through all un-labeled rows!
ROW_LIMIT = None  


class PostClassification(BaseModel):
    id: str
    label: str = Field(validation_alias=AliasChoices('label', 'category'))
    rationale: str = "" 
    title: Optional[str] = None

class BatchClassificationResponse(BaseModel):
    classifications: List[PostClassification]


# System Prompt updated to use Community & Culture instead of Fan Creations
SYSTEM_PROMPT = """
You are an expert data classification bot specialized in text categorization for a music fan community (r/TaylorSwift).
Your task is to classify a batch of Reddit post titles into exactly one of four categories: "Updates", "Community & Culture", "Poll", or "Opinion".

CRITERIA DEFINITIONS:
1. Updates: Commercial/chart milestones, official merchandise drops, brand announcements, mainstream media coverage, official studio/tour footage, or organized real-world fan events.
2. Community & Culture: Fan-centric items celebrating community identity. Includes user-generated digital/physical art, lyric/song mashups, vocal or instrument covers, tattoo mockups, custom font collections, and detailed fan-made concept projects.
3. Poll: Open-ended community discussion questions designed to prompt varied personal replies, favorites, rankings, or hypothetical scenarios from general users.
4. Opinion: Long-form reviews, deep-dive lyrical explanations, structural analyses, essays on writing habits, or thematic arguments.

You must return a valid JSON object matching the requested schema. Do not include markdown code fences or conversational filler.
"""

def process_groq_batch(batch_posts: List[dict]) -> dict:
    user_content = f"Classify the following batch of post titles:\n{json.dumps(batch_posts, indent=2)}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        raw_output = completion.choices[0].message.content
        response_dict = json.loads(raw_output)
        parsed_response = BatchClassificationResponse.model_validate(response_dict)
        
        batch_map = {}
        for item in parsed_response.classifications:
            batch_map[str(item.id).strip()] = (item.label.strip(), item.rationale.strip())
        return batch_map
        
    except Exception as e:
        print(f"\n[Warning] API validation failed for batch: {e}")
        return {}

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Target file '{INPUT_CSV}' not found.")
        return

    df = pd.read_csv(INPUT_CSV)

    # Initialize columns if they got wiped or are missing
    for col in ['label', 'ai_prelabeled', 'manually_verified']:
        if col not in df.columns:
            df[col] = False if col == 'manually_verified' or col == 'ai_prelabeled' else ""
            
    df['label'] = df['label'].fillna("").astype(str).str.strip()
    df['manually_verified'] = df['manually_verified'].fillna(False).astype(bool)

    # Target ONLY unverified, unlabeled rows so you can safely resume anytime
    valid_labels = ["Updates", "Community & Culture", "Poll", "Opinion"]
    unlabeled_mask = (~df['label'].isin(valid_labels)) & (df['manually_verified'] == False)
    unlabeled_df = df[unlabeled_mask]
    
    if len(unlabeled_df) == 0:
        print("🎉 No unlabeled rows found! Everything has a category tag.")
        return
        
    print(f"🚀 Found {len(unlabeled_df)} rows waiting for title classification.")
    
    # Structure payload list (Back to ONLY id and title)
    posts_to_process = unlabeled_df[['id', 'title']].to_dict(orient='records')
    if ROW_LIMIT is not None:
        posts_to_process = posts_to_process[:ROW_LIMIT]

    # Process and commit immediately batch-by-batch
    for i in tqdm(range(0, len(posts_to_process), BATCH_SIZE), desc="Labeling Batches"):
        batch = posts_to_process[i:i + BATCH_SIZE]
        batch_results = process_groq_batch(batch)
        
        if not batch_results:
            continue
            
        for post_id, (predicted_label, rationale) in batch_results.items():
            if predicted_label in valid_labels:
                idx_match = df[df['id'] == post_id].index
                if not idx_match.empty:
                    df.at[idx_match[0], 'label'] = predicted_label
                    df.at[idx_match[0], 'ai_prelabeled'] = True
                    
        # Flush progress to CSV instantly
        df.to_csv(INPUT_CSV, index=False, encoding='utf-8')
        time.sleep(1.0)

    print(f"\n✅ Done! Progress saved directly to {INPUT_CSV}")

if __name__ == "__main__":
    main()