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
   * *Example 1*: "Taylor Swift spotted arriving at Electric Lady Studios in NYC last night, sparking fresh album rumors."
   * *Example 2*: "Billboard confirms her latest single has officially debuted at #1 on the Hot 100 chart."

2. **`Community & Culture`**: First-person sharing of fan-made art, crafts, digital projects, or real-world lifestyle experiences and encounters related to being a fan.
   * *Example 1*: "I finally finished crocheting my custom cardigan just in time for the upcoming show!"
   * *Example 2*: "Look at this adorable Taylor Swift themed cafe and shop I stumbled across on my trip today!"

3. **`Poll`**: Interactive community games, elimination brackets, tournaments, or explicit requests forcing the community to vote, rank, or choose between options.
   * *Example 1*: "Round 4: Which track from the Folklore era are we eliminating next? Vote in the comments."
   * *Example 2*: "Rank your top 3 favorite track-five bridges from her entire discography."

4. **`Opinion`**: Monologue-style text posts sharing an analysis, personal perspective, defense, or critique of the music, lyrics, or culture without prompting a structured vote.
   * *Example 1*: "I honestly think her lower register vocals on this album are completely underrated and don't get enough credit."
   * *Example 2*: "A deep dive into the literary and historical references scattered throughout the new tracks."

---

## 3. Hard Edge Cases

### Opinionated Questions vs. Polls
The most frequent edge case occurs when a user writes a multi-paragraph personal analysis (**Opinion**) but ends the post title or body text with an open-ended conversational question (e.g., *"I think Album X is her undisputed masterpiece. What do you guys think?"*). Both `Opinion` and `Poll` contain subjective thoughts, but their interactive intent differs.

To handle this consistently during annotation, we will use the following to help us decide:
* If the post text explicitly forces a structural choice, ranking, or data-driven metric (*"Pick one," "Rank these 5 songs," "Vote on this bracket"*), it must be labeled a **`Poll`**. 
* If the question at the end is more of an invitation for casual conversation or open validation (*"Does anyone else agree?", "What are your thoughts?"*), the primary volume of text remains a personal monologue, and it will be labeled an **`Opinion`**.

### News-Driven Opinion
This is an example that would generally start with a real-world event (which would sound like **`Update`**), and then shifts to focus on an analysis of this event (which would sound like **`Opinion`**). For example, "Pitchfork gave the album a 6.5. Personally, I think the reviewer completely missed the emotional depth of the track-five bridges."

To handle this, we will use the following to help us decide: if the objective news item is used purely as a catalyst for a personal rant or lyrical critique, label it **`Opinion`**. It should only be labeled an **`Updates`** post if the primary utility of the text is to share raw data or official links with minimal user commentary.

---

## 4. Data Collection Plan
* **Source Location**: Data will be extracted directly from `r/TaylorSwift` using browser JSON endpoints (`/new/.json` and `/top/.json`).
* **Target Volume**: The project will build a dataset of 200 posts, targetting a balanced distribution of **50 examples per label**.

### Underrepresented Label Mitigation Strategy
Because the subreddit naturally defaults to an excess of `Opinion` posts, a consecutive scrape of the "Hot" page will result in severe class imbalance. If a category like `Community & Culture` or `Poll` remains underrepresented after the initial 200-post pull, we will execute targeted mining. 
1. We will filter the spreadsheet data using Reddit's user-assigned flairs (such as filtering specifically for the *Fanmade* or *Merch* flair to find community culture posts).
2. We will deliberately pull from the monthly "Top" filter to find highly upvoted community bracket games to fill out the `Poll` requirement. 
3. Excess `Opinion` posts will be systematically pruned from the final training array until the distribution scales evenly back to a balanced baseline.

---

## 5. Evaluation Metrics
Evaluating this multi-class classifier requires looking deeper than overall accuracy, as accuracy can hide critical systematic errors across individual categories. We will implement three specific metrics:

* **Macro-Averaged F1-Score**: This is the primary metric. The F1-Score calculates the harmonic mean of Precision (avoiding false positives) and Recall (avoiding false negatives). Using a *macro* average ensures that all four categories are weighted equally in the final score, guaranteeing the model is truly performing well across all buckets rather than masking poor minority-class performance behind a single high-volume category.
* **Confusion Matrix Visualization**: We will plot a multi-class confusion matrix to spot specific linguistic leakages. For example, it is vital to track if the model is routinely misclassifying casual lifestyle updates within `Community & Culture` as official journalistic entries under `Updates`.
* **Per-Class Precision**: Crucial for testing the utility of the `Updates` category. If a community moderation tool relies on this model to auto-filter news, we need high precision in that specific class to ensure regular fan opinions aren't accidentally suppressed under an "official update" flag.

---

## 6. Definition of Success
For this specialized 4-class domain task, a baseline **Macro F1-Score of $\ge 0.80$ (80%)** will serve as our definition of a successful, deployable model.

### Why this is the target
In a 4-label ecosystem, a completely random guess yields a 25% accuracy rate. Achieving an 80% F1-score means the DistilBERT model has successfully bypassed simple word-matching and is accurately parsing complex human structures—such as separating a fan's personal reaction to an event from the news event itself. 

At $\ge 0.80$, the model is genuinely useful as a community tool. It is reliable enough to run in the background of a subreddit to auto-categorize incoming feeds, power custom user filters (e.g., allowing a user to hide all `Polls` but view all `Community & Culture` projects), and assist human moderators by organizing massive influxes of data during major career events.

## AI Tool Plan

### 1. Label Stress-Testing
To ensure our taxonomy definitions were robust before annotating our 200 scraped examples, we used Gemini to generate highly ambiguous edge cases that sit right on the boundaries between our categories. Reviewing these specific posts allowed us to lock down our annotation rules before processing the data:

*   **Boundary Post 1:** *"Pitchfork gave the album a 6.5. Personally, I think the reviewer completely missed the emotional depth of the track-five bridges."*
    *   *The Conflict:* **`Updates` vs. `Opinion`**
    *   *Resolution:* **`Opinion`**. The objective news item is used purely as a catalyst for a personal rant. The primary volume of text relies on subjective phrasing (*"Personally, I think"*, *"missed the emotional depth"*).
*   **Boundary Post 2:** *"I built a web scraper that tracks her Billboard streaming metrics and auto-tweets the daily changes! Let me know what features you want me to add next."*
    *   *The Conflict:* **`Updates` vs. `Community & Culture`**
    *   *Resolution:* **`Community & Culture`**. Because a member of the fandom personally built this tool and is showcasing their own labor, it belongs here. 
*   **Boundary Post 3:** *"I was so inspired by her Electric Lady Studios street style that I spent all weekend sewing a replica of that brown leather vest. Check out my final look!"*
    *   *The Conflict:* **`Updates` vs. `Community & Culture`**
    *   *Resolution:* **`Community & Culture`**. The user relies heavily on first-person possessive pronouns (*"I spent"*, *"my final look"*) to showcase a physical craft project. The paparazzi sighting is just context.
*   **Boundary Post 4:** *"Which song do you think is her most lyrically complex? I’ve written a 3-paragraph defense of 'Happiness' in the body text, but I want to see your arguments below."*
    *   *The Conflict:* **`Opinion` vs. `Poll`**
    *   *Resolution:* **`Opinion`**. The question in the title is an open-ended conversational invitation. Because the author's primary intent is to share their own monologue essay, it clusters semantically with `Opinion`.
*   **Boundary Post 5:** *"The official store just dropped the signed vinyl restocks! Did anyone actually manage to check out before they sold out, or did we all get stuck in the queue?"*
    *   *The Conflict:* **`Updates` vs. `Opinion`**
    *   *Resolution:* **`Updates`**. Retail drop alerts provide critical, objective community utility. Even though the user appends a frustrated personal question, the core linguistic footprint relies on objective retail alerts (*"store dropped"*, *"vinyl restocks"*, *"sold out"*).
*   **Boundary Post 6:** *"Let’s settle this once and for all: Tier List of every song where she sings about rain. Let me know your top tier in the comments!"*
    *   *The Conflict:* **`Poll` vs. `Opinion`**
    *   *Resolution:* **`Poll`**. The text explicitly forces a data-driven categorization and structural constraints (*"Tier List"*, *"settle this"*, *"top tier"*), making it a structured community game rather than a standard monologue.

### 2. Annotation Assistance
To accelerate dataset creation without compromising ground-truth integrity, **Groq (Llama-3.3-70b-versatile)** will be used to generate a baseline layer of "soft labels" for our 200 posts.
*   **Workflow:** A Python script will pass the title and body text of each scraped post to the Groq API alongside a system prompt containing our 4-label taxonomy rules. The model will output an initial prediction.
*   **Tracking and Verification:** Inside our master `taylor_swift_dataset.csv` file, a metadata column titled `ai_prelabeled` (Boolean) will track every instance where a soft label was generated. 100% of these rows will undergo a manual human validation audit. If the LLM prediction matches our playbook rules, it will be locked in; if it fails an edge-case rule, it will be manually overwritten to guarantee a flawless training and test split.

### 3. Failure Analysis
Post-training, instances where our fine-tuned **DistilBERT** model's predictions disagree with the validation ground truth will be collected into a dedicated error log DataFrame.
*   **AI Diagnostics:** This array of incorrect rows (containing text, true label, and wrong prediction) will be packaged and sent to the **Llama-3.3-70b** model via Groq. The model will be prompted to run a thematic analysis to identify hidden semantic patterns behind the failures (e.g., identifying if DistilBERT is over-indexing on specific artist name proper nouns inside `Community & Culture` and misclassifying them as `Updates`).
*   **Human Verification:** Any pattern flagged by the AI will be manually audited by pulling 3 specific examples from that failure cluster to verify if the linguistic pattern is systematically reproducible before completing the final project evaluation write-up.