# Project Planning: Taylor Swift Subreddit Classifier

The goal of this project is to go through the process of identifying, labeling and creating an annotated dataset reflective of an online community. The second goal of this project is to use this dataset to train `distilbert-base-uncased` on the labeled dataset.

## 1. Community
**Selected Subreddit:** `r/TaylorSwift`

### Why this Community?
The `r/TaylorSwift` community is an incredibly active, dedicated digital space where millions of fans are able to gather to share updates, opinions, and their own personal connections to Taylor Swift and her career. Taylor has an enormous influence on the music industry and the lives of many of her fans - this deeply personal connection also means that the community is active and that the subreddit also has highly diverse types of posts.

In addition to this being a great dataset to work with, I've been a fan of Taylor's for almost 2 decades, so I also have a personal connection to this topic which is why I decided to choose this community!

### Suitability for Classification
This community is a perfect fit for a text classification task because its discourse has a lot of variety. For example, there are many posts that read like professional media alerts/standard journalism, some posts focus around personal arts/crafts that people have made to engage with Taylor and her music, and others focus on asking sharing opinions on her songs or career choices.

---

## 2. Labels
The taxonomy for this dataset consists of four categories:

### Label Definitions and Examples

1. **`Updates`**: Third-person objective reporting of real-world facts, industry news, career metrics, or media sightings.
   * *Example 1*: "Taylor Swift officially announces '1989 (Taylor's Version)' will be released on October 27th, 2023."
   * *Example 2*: "Billboard confirms her latest single has officially debuted at #1 on the Hot 100 chart."

2. **`Community`**: First-person sharing of fan-made art, crafts, digital projects, or real-world lifestyle experiences and encounters related to being a fan.
   * *Example 1*: "I spent all weekend making these custom friendship bracelets for the show in London and I’m so happy with how they turned out!"
   * *Example 2*: "Look at this adorable Taylor Swift themed cafe and shop I stumbled across on my trip today!"

3. **`Poll`**: Interactive community games, elimination brackets, tournaments, or explicit requests forcing the community to vote, rank, or choose between options.
   * *Example 1*: "RANKED: reputation Album Hey guys!! It's me, u/itookyourmatches, and I'm back again with the annual track elimination game! Vote for your least favorite song in the Google Form below, and tomorrow we will reveal who got knocked out of the top spot."
   * *Example 2*: "Rank your top 3 favorite track-five bridges from her entire discography."

4. **`Opinion`**: Monologue-style text posts sharing an analysis, personal perspective, defense, or critique of the music, lyrics, or culture without prompting a structured vote.
   * *Example 1*: "I honestly think her lower register vocals on this album are completely underrated and don't get enough credit."
   * *Example 2*: "A deep dive into the literary and historical references scattered throughout the new tracks."

---

## 3. Hard Edge Cases

### Opinionated Questions vs. Polls
The most frequent edge case occurs when a user writes a multi-paragraph personal analysis (**Opinion**) but ends the post title or body text with an open-ended conversational question (e.g., *"I think Album X is her undisputed masterpiece. What do you guys think?"*). Both `Opinion` and `Poll` contain subjective thoughts, but their interactive intent differs.

To handle this consistently during annotation, we use the following to help us decide:
* If the post text explicitly forces a structural choice, ranking, or data-driven metric (*"Pick one," "Rank these 5 songs," "Vote on this bracket"*), it must be labeled a **`Poll`**. 
* If the question at the end is more of an invitation for casual conversation or open validation (*"Does anyone else agree?", "What are your thoughts?"*), the primary volume of text remains a personal monologue, and it will be labeled an **`Opinion`**.

* **Opinion Example**: *"Debut Taylor’s Version concept I don’t love the outfit she has on in the original photo. I think this is a nice callback to Debut with the white dress."* (Row 2). 
* **Poll Example**: *"RANKED: reputation Album Hey guys!! It's me, u/itookyourmatches, and I'm back again with the annual ..."* (Row 186).

### News-Driven Opinion
This occurs when a post starts with a real-world event (which would sound like **`Update`**), and then shifts to focus on an analysis of this event (which would sound like **`Opinion`**). 

To handle this, we use the following to help us decide: if the objective news item is used purely as a catalyst for a personal rant or lyrical critique, label it **`Opinion`**. It should only be labeled an **`Updates`** post if the primary utility of the text is to share raw data or official links with minimal user commentary.

* **Update Example**: *"13 bits of wisdom from Taylor Swift's Songwriters Hall of Fame speech"* (Row 88). Even though the user adds framing, the core value is the dissemination of official verbatim quotes, making it an **`Updates`** post.

### Fan-Culture, Memes, and Meta-Content
We encountered several posts that initially appeared to be **`Opinion`** or **`Community`** but required deeper classification based on their functional intent within the subreddit ecosystem.

To handle these consistently, we use the following decision framework:
* **Fan-Participation Games**: Posts that invite the community to engage with a specific, time-sensitive, or repetitive external game are labeled **`Poll`** if they involve a structured call-to-action, or **`Community`** if they are primarily celebratory/humorous.
    * *Example*: *"Top spot for the taking for any true Swiftie on today's Pixle..."* (Row 191). This is **`Community`** because the primary utility is facilitating shared communal engagement rather than critiquing content.
* **Low-Effort Humorous/Reference Posts**: Posts consisting almost entirely of a catchphrase, a meme, or a niche reference are labeled **`Community`**. 
    * *Example*: *"There's a snake in my boot 🐍🦋"* (Row 148). This is **`Community`** because it is a low-effort, symbolic fan reference that lacks analytical substance.

---

## 4. Data Collection Plan
* **Source Location**: Data will be extracted directly from `r/TaylorSwift` using browser JSON endpoints (`/new/.json` and `/top/.json`).
* **Target Volume**: The project will build a dataset of 200 posts, targetting a balanced distribution of **50 examples per label**.

### Underrepresented Label Mitigation Strategy
Because the subreddit naturally defaults to an excess of `Opinion` posts, a consecutive scrape of the "Hot" page will result in severe class imbalance. If a category like `Community` or `Poll` remains underrepresented after the initial 200-post pull, we will execute targeted mining. 
1. We will filter the spreadsheet data using Reddit's user-assigned flairs (such as filtering specifically for the *Fanmade* or *Merch* flair to find community posts).
2. We will deliberately pull from the monthly "Top" filter to find highly upvoted community bracket games to fill out the `Poll` requirement. 
3. Excess `Opinion` posts will be systematically pruned from the final training array until the distribution scales evenly back to a balanced baseline.

---

## 5. Evaluation Metrics
Evaluating this multi-class classifier requires looking deeper than overall accuracy, as accuracy can hide critical systematic errors across individual categories. We will implement three specific metrics:

* **Averaged F1-Score**: This is the primary metric. The F1-Score calculates the harmonic mean of Precision (avoiding false positives) and Recall (avoiding false negatives). Using an average ensures that all four categories are weighted equally in the final score, guaranteeing the model is truly performing well across all buckets rather than masking poor minority-class performance behind a single high-volume category.
* **Confusion Matrix Visualization**: We will plot a multi-class confusion matrix to spot specific linguistic leakages. For example, it is vital to track if the model is routinely misclassifying casual lifestyle updates within `Community` as official journalistic entries under `Updates`.
* **Per-Class Precision**: Crucial for testing the utility of the `Updates` category. If a community moderation tool relies on this model to auto-filter news, we need high precision in that specific class to ensure regular fan opinions aren't accidentally suppressed under an "official update" flag.

---

## 6. Definition of Success
For this specialized 4-class domain task, a baseline **Macro F1-Score of $\ge 0.80$ (80%)** will serve as our definition of a successful, deployable model.

### Why this is the target
In a 4-label ecosystem, a completely random guess yields a 25% accuracy rate. Achieving an 80% F1-score means the DistilBERT model has successfully bypassed simple word-matching and is accurately parsing complex human structures—such as separating a fan's personal reaction to an event from the news event itself. 

At $\ge 0.80$, the model is genuinely useful as a community tool. It is reliable enough to run in the background of a subreddit to auto-categorize incoming feeds, power custom user filters (e.g., allowing a user to hide all `Polls` but view all `Community` projects), and assist human moderators by organizing massive influxes of data during major career events.

## AI Tool Plan

### 1. Data Collection
I used Gemini to write a script to scrape data directly from Reddit. That script is in `collect_data.py`.

### 2. Label Stress-Testing
To ensure our taxonomy definitions were robust before annotating our 200 scraped examples, we used Gemini to generate highly ambiguous edge cases that sit right on the boundaries between our categories. Reviewing these specific posts allowed us to lock down our annotation rules before processing the data:

*   **Boundary Post 1:** *"Pitchfork gave the album a 6.5. Personally, I think the reviewer completely missed the emotional depth of the track-five bridges."*
    *   *The Conflict:* **`Updates` vs. `Opinion`**
    *   *Resolution:* **`Opinion`**. The objective news item is used purely as a catalyst for a personal rant. The primary volume of text relies on subjective phrasing (*"Personally, I think"*, *"missed the emotional depth"*).
*   **Boundary Post 2:** *"I built a web scraper that tracks her Billboard streaming metrics and auto-tweets the daily changes! Let me know what features you want me to add next."*
    *   *The Conflict:* **`Updates` vs. `Community`**
    *   *Resolution:* **`Community`**. Because a member of the fandom personally built this tool and is showcasing their own labor, it belongs here. 
*   **Boundary Post 3:** *"I was so inspired by her Electric Lady Studios street style that I spent all weekend sewing a replica of that brown leather vest. Check out my final look!"*
    *   *The Conflict:* **`Updates` vs. `Community`**
    *   *Resolution:* **`Community`**. The user relies heavily on first-person possessive pronouns (*"I spent"*, *"my final look"*) to showcase a physical craft project. The paparazzi sighting is just context.
*   **Boundary Post 4:** *"Which song do you think is her most lyrically complex? I’ve written a 3-paragraph defense of 'Happiness' in the body text, but I want to see your arguments below."*
    *   *The Conflict:* **`Opinion` vs. `Poll`**
    *   *Resolution:* **`Opinion`**. The question in the title is an open-ended conversational invitation. Because the author's primary intent is to share their own monologue essay, it clusters semantically with `Opinion`.
*   **Boundary Post 5:** *"The official store just dropped the signed vinyl restocks! Did anyone actually manage to check out before they sold out, or did we all get stuck in the queue?"*
    *   *The Conflict:* **`Updates` vs. `Opinion`**
    *   *Resolution:* **`Updates`**. Retail drop alerts provide critical, objective community utility. Even though the user appends a frustrated personal question, the core linguistic footprint relies on objective retail alerts (*"store dropped"*, *"vinyl restocks"*, *"sold out"*).
*   **Boundary Post 6:** *"Let’s settle this once and for all: Tier List of every song where she sings about rain. Let me know your top tier in the comments!"*
    *   *The Conflict:* **`Poll` vs. `Opinion`**
    *   *Resolution:* **`Poll`**. The text explicitly forces a data-driven categorization and structural constraints (*"Tier List"*, *"settle this"*, *"top tier"*), making it a structured community game rather than a standard monologue.

### 3. Annotation Assistance
To accelerate dataset creation without compromising ground-truth integrity, Groq (Llama-3.3-70b-versatile) was used to generate a baseline layer of "soft labels" for the 200 posts. I used Gemini to write the scripts referenced here:
*   **Workflow:** A Python script passes the text of each scraped post to the Groq API alongside a system prompt containing the 4-label taxonomy rules. The model output an initial prediction. 
*   **Tracking and Verification:** Inside our master `taylor_swift_dataset.csv` file, a metadata column titled `ai_prelabeled` (Boolean) tracked every instance where a soft label was generated. 100% of these rows underwent a manual validation. Where the LLM's classification was ambiguous or failed to align with our specific rules, it was manually overwritten. For complex cases where the reasoning behind a label change was particularly nuanced, those details have been documented in the notes column of the final dataset to ensure full transparency and reproducibility. I used Gemini to write a script to aid in the manual audit of each of the labels - that's in `validate_labels.py`. I also used Gemini to convert this main file into a new file which is the final one that is formatted (text, label, notes) as per what the Google Notebook for this project requires.
*   Assessing results: I used Gemini to write a script to count how many rows were attributed per label to make sure no one label had the majority of labels.

## Iterative Refinement and Training Adjustments
Training the distilbert-base-uncased model was an iterative process of hypothesis testing. We adjusted our strategy based on real-time observations of model performance and data quality. The following adjustments were made to address specific challenges:

* Dataset Expansion for Semantic Depth: After the initial training, we observed that the "Opinion" label had poor recall because the model was failing to capture the nuance of long-form analytical posts. We expanded the dataset by adding 50+ entries specifically targeting argumentative posts (taylor_swift_labeled_data_final_updated_with_addnl_data.csv). While this initially increased "noise," it was a necessary step to broaden the model's exposure to complex linguistic structures.

* Sequence Length Optimization: Our error analysis revealed that the model was struggling with longer "Opinion" posts. We hypothesized that the default sequence length of 256 tokens was truncating the logical arguments contained in the posts, causing the model to lose context. We increased the max_seq_length to 512 tokens. This allowed the model to ingest the full "argument structure" (thesis, evidence, and rhetorical conclusion) rather than just the first few sentences.

* Hyperparameter Stabilization: Following the dataset expansion, we observed signs of overfitting to the smaller training set. To stabilize the training process, we decreased the learning rate to 5e-6.