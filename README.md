# TakeMeter: Evaluation Report

## 1. Project Summary
TakeMeter is a text classification system built for the `r/TaylorSwift` subreddit. It categorizes content (posts from the subreddit) into four labels: **Updates**, **Opinion**, **Community**, and **Poll**. The goal was to build a system capable of distinguishing between objective news, subjective analysis, creative community projects, and posts that are meant to spark responses to a question or questions. We compared a baseline heuristic approach (zero-shot) (using a prompt-based LLM) against a fine-tuned `distilbert-base-uncased` model. The training platform is HuggingFace.

## 2. Methodology
The development of the TakeMeter classifier with the fine-tuned model was an iterative process driven by continuous error analysis and hypothesis testing. Rather than relying on a static pipeline, the model underwent several rounds of refinement to address label imbalance and semantic overlap.

We began by creating a balanced dataset of 200 posts from r/TaylorSwift across four labels: Updates, Opinion, Community, and Poll.  The `r/TaylorSwift` community is an incredibly active, dedicated digital space where millions of fans are able to gather to share updates, opinions, and their own personal connections to Taylor Swift and her career. Taylor has an enormous influence on the music industry and the lives of many of her fans - this deeply personal connection also means that the community is active and that the subreddit also has highly diverse types of posts. That is why we selected this subreddit.

### Data Collection

The final dataset used is in `taylor_swift_labeled_data_final_updated_with_addnl_data.csv`.

The dataset consists of 216 (initially 200 but we increased to 216) posts scraped from Reddit using a custom Python script (collect_data.py). The objective was to curate a representative sample of fan discourse to train a classifier capable of distinguishing between updates, community engagement, opinions, and poll-based interaction.

To ensure high data integrity, we employed a hybrid "human-in-the-loop" annotation strategy:
* Taxonomy Definitions: We established a four-label taxonomy (Updates, Community, Opinion, Poll).
* Soft Labeling: An initial baseline of "soft labels" was generated using Llama-3.3-70b-versatile via the Groq API.
* Manual Audit: Every entry was manually reviewed. We utilized an ai_prelabeled metadata column to track all machine-assisted predictions. I conducted a 100% manual audit of the entire dataset, overriding AI labels where they deviated from our taxonomy rules to ensure the ground truth was determined by human judgment.
* Verification Tools: Scripts such as validate_labels.py were used to audit label distribution and ensure consistent formatting for the project requirements.

This is how the four labels were defined:
1. **`Updates`**: Third-person objective reporting of real-world facts, industry news, career metrics, or media sightings.
   * *Example 1*: "Chinese swifties host a live megamix of Taylor’s greatest hits in celebration her 20th anniversary in Chengdu."
   * *Example 2*: "Japanese music magazine INROCK made a special feature about "The 20-Year Journey with Taylor and Our Passionate Feelings" sent in from fans in Japan."

2. **`Community`**: First-person sharing of fan-made art, crafts, digital projects, or real-world lifestyle experiences and encounters related to being a fan.
   * *Example 1*: ""some of my favourite redesigned/textless album art i've made more taylor cover art and folders here 1-2: debut deluxe 3: fearless love story 4: fearless you belong with me 5: speak now back to december 6-8: speak now textless, target edition, and single 9: beautiful ghosts 10: you need to calm down 11: fearless tv colourised 12: 1989 tv sunset blvd 13: lavender haze 14: speak now tv recoloured like the target edition of the first one. 15: tortured poets department i can do it with a broken heart 16: tortured poets department but daddy i love him edition colourised 17: the tortured poets department anthology vinyl colourised 18: the fate of ophelia 20: the life of a showgirl opalite"
   * *Example 2*: "Taylor Swift themed cafe and shop! My partner knows I'm a huge fan so for my birthday took me to Beaumaris in North Wales to visit the Mock Turtle shop and August cafe. The whole cafe is Taylor themed with Taylor drinks (the Menu script), decor, music, and cookies. There was also a shop with mountains of Taylor themed gifts! I spent way too much money but got socks, a Taylor Stanley, gifts for friends etc"

3. **`Poll`**: Interactive community games, elimination brackets, tournaments, or explicit requests forcing the community to vote, rank, or choose between options.
   * *Example 1*: "Could you make a hit album with just vault tracks? Does anybody else think that you could make an album made of just Taylor Swift vault tracks that would be a hit album for any other artist? If so, what would be your theme and track list?"
   * *Example 2*: "what are the most relatable songs? what are your most relatable taylor swift songs for any emotion?"

4. **`Opinion`**: Monologue-style text posts sharing an analysis, personal perspective, defense, or critique of the music, lyrics, or culture without prompting a structured vote.
   * *Example 1*: "Debut Taylor’s Version concept I don’t love the outfit she has on in the original photo. I think this is a nice callback to Debut with the white dress."
   * *Example 2*: "So Long London made me see my breakup differently Listening to So Long London by Taylor Swift brought something into focus for me that I have been circling for a while but not fully naming. It was the line about how much sadness do you think I had in me that made me realise how much of my four year relationship had become about endurance rather than mutual care. I do not see my ex as a villain. I do not think he is a bad person. I think we ended up in something where the structure of it allowed him to benefit from my lack of boundaries and where I slowly made his emotional world the centre of mine. In doing that I stopped really being present in my own life. There was no dramatic explosion at the end. It was quieter than that. When it ended I expected the kind of grief I had always known after previous relationships, the kind that feels like being completely undone. Instead there was sadness, but also something I had not experienced before which was relief. Not relief at losing someone I did not care about, but relief at no longer being in something that required me to disappear in order to hold it together. I do not feel jealousy about where he is now. I do not feel comparison with his new relationship. If anything I feel a strange distance from all of that. I can see she seems like a lovely person and I genuinely wish her well, but I also find myself hoping she does not end up in the same dynamic I was in. What I am left with is the recognition that I learned how easily I can abandon myself when I am focused on someone else. His feelings and struggles became my whole world and somewhere along the way I stopped existing as a separate person in it. If there is anything I am taking forward, it is that love cannot be the place where I hide from myself. I do not need to withstand that much to be worthy of connection. And I do not need to make someone else emotional life my responsibility in order to matter. It is not bitterness. It is not blame. It is just clarity."

Here were some examples that were difficult to classify:
* "The official store just dropped the signed vinyl restocks! Did anyone actually manage to check out before they sold out...?" 
* * While the user includes a conversational question, the post’s primary utility is providing critical, objective retail information (store/restock status). The objective "Update" content overrides the personal tone.

* "Which song do you think is her most lyrically complex? I’ve written a 3-paragraph defense of 'Happiness' in the body text..."
* * Although the title is phrased as a question, the body text is a structured monologue. Because the author's primary intent is to share an essay, the "Opinion" label best captures the semantic weight of the post.

* "I built a web scraper that tracks her streaming metrics and auto-tweets the daily changes! Let me know what features you want me to add next."
* * This was classified as "Community" rather than "Updates" because it showcases individual fan labor and creative contribution to the fandom, rather than acting as a standard news feed.

### Zero Shot Attempt (Baseline)
To establish a performance baseline for our classification task, we utilized Llama-3.3-70b-versatile (via Groq). We employed a zero-shot prompting strategy, providing the model with precise definitions for each of our four target categories. 

Here is the system prompt we used:

```
You are classifying Reddit posts from r/TaylorSwift subreddit.
Assign each post to exactly one of the following categories.

Updates: Posts that share raw data, official links, news, or public milestones with minimal user commentary.
Example: "Taylor Swift officially announces '1989 (Taylor's Version)' will be released on October 27th, 2023."

Community: Celebratory posts, memes, low-effort fan references, or general community engagement that lacks critical analysis.
Example: "I spent all weekend making these custom friendship bracelets for the show in London and I’m so happy with how they turned out!"

Poll: Posts that explicitly force a structural choice, ranking, or data-driven metric from the community.
Example: "RANKED: reputation Album Hey guys!! It's me, u/itookyourmatches, and I'm back again with the annual track elimination game! Vote for your least favorite song in the Google Form below, and tomorrow we will reveal who got knocked out of the top spot."

Opinion: Personal monologues, lyrical critiques, or subjective aesthetic analyses where the user expresses their own viewpoint.
Example: "Debut Taylor’s Version concept I don’t love the outfit she has on in the original photo. I think this is a nice callback to Debut with the white dress."

Respond with ONLY the exact label name (Updates, Community, Poll, or Opinion). 
Do not include any other text, punctuation, or reasoning.

Valid labels:
Updates
Community
Poll
Opinion

```

### Fine Tuning Attempt

We used `distilbert-base-uncased` as the base model.

After initial training and observing that the performance of the fine-tuned model was almost as good but not better than the zero-shot approach, we observed that the "Opinion" label was significantly underrepresented in terms of its semantic breadth, leading to poor recall. 

To address the poor performance of the "Opinion" label, we expanded the dataset by adding more entries specifically targeting argumentative posts. The file with additional entries is titled `taylor_swift_labeled_data_final_updated_with_addnl_data.csv`. Counter-intuitively, this resulted in a decrease in overall accuracy. Error analysis revealed that the new "Opinion" examples were semantically noisy, overlapping heavily with "Updates" and "Community" posts, which confused the model’s classification boundary.

We hypothesized that the model was truncating critical context within the longer, more complex "Opinion" posts. To test this, we increased the sequence length from the default 256 to 512. This change was intended to allow the model to ingest more of the argument structure of each post.

Following the sequence length adjustment, we performed hyperparameter tuning to stabilize the training process. We specifically focused on decreasing the learning rate (to 5e-6) to prevent the model from overshooting during weight updates and to encourage better convergence on the smaller dataset. 

Despite these refinements, the model’s performance eventually plateaued. The final evaluation suggests that the current performance limit is dictated by the inherent semantic entanglement of the chosen labels within this specific community, rather than model architecture constraints. This plateau indicates that further improvement would require a shift in taxonomy or significantly higher data quality, rather than further hyperparameter manipulation.

## 3. Evaluation Results and Confusion Matrix

Here are the evaluation results:

{
  "baseline_accuracy": 0.6364,
  "finetuned_accuracy": 0.6061,
  "improvement": -0.0303,
  "test_set_size": 33,
  "label_map": {
    "Updates": 0,
    "Opinion": 1,
    "Community": 2,
    "Poll": 3
  },
  "model": "distilbert-base-uncased"
}

### Fine-Tuned Model Confusion Matrix

| True Label \ Predicted Label | Updates | Opinion | Community | Poll |
| :--- | :---: | :---: | :---: | :---: |
| **Updates** | 9 | 0 | 0 | 1 |
| **Opinion** | 1 | 0 | 0 | 4 |
| **Community** | 3 | 0 | 2 | 4 |
| **Poll** | 0 | 0 | 0 | 9 |

#### Most Confused Pair: 
The most significant source of confusion is between Community (actual) and Poll (predicted), which accounts for 4 errors. Additionally, there is a notable confusion between Opinion (actual) and Poll (predicted), which accounts for another 4 errors.

#### Directional Pattern: 
The model shows a strong directional bias toward misclassifying both "Community" and "Opinion" posts as "Poll" posts.

#### Model Boundary:
The boundary the model has not learned is how to differentiate between content that might contain community engagement or opinionated discussion and content that is strictly a "Poll." It appears that the model is over-predicting the "Poll" label when faced with "Community" or "Opinion" content, suggesting it may be misinterpreting the interactive or inquisitive nature of these posts as polling behavior.

The difficulty in establishing a clear boundary between Poll, Community, and Opinion labels, as evidenced by the misclassifications in "confusion_matrix (2).png", likely stems from structural similarities rather than just ambiguity or sarcasm.

Based on the nature of the posts, the model is likely struggling with the following factors:

* Interrogative "Call to Action": Both the Opinion and Community examples frequently conclude with open-ended questions designed to foster engagement (e.g., "What are your thoughts on it now?", "Would love to hear what your opinions are... and if you’d change/replace/add anything?", or "What's your ideal triple album?"). These structures closely mimic the direct, question-based format of a Poll.

* Structural Mimicry: While Opinion posts are often longer, wordy, and analytical, the presence of these concluding questions likely triggers the model to classify them as "Polls" because it identifies the interrogative structure as a stronger signal than the surrounding long-form prose.

* Shared Purpose: Community posts often share a "creative" or "curated" project and then invite others to participate or critique it. This blend of long-form content sharing (Community) with a direct request for input (Poll) creates a hybrid structure where the topic signals one label (e.g., a personal project) but the structure (the request for input) signals another (e.g., a Poll)

In short, the model appears to be over-weighting the grammatical structure of the final sentences - which act as engagement prompts—rather than analyzing the entire body of the text to distinguish between an opinionated essay, a community-driven project, or a simple polling question.

The issue is likely a data problem rooted in the training data distribution rather than simple annotation inconsistency. The labels Opinion and Community often utilize an "engagement-first" structure. When a post is labeled as an Opinion or Community piece, but features a strong closing question that reads like a survey, the model encounters a structural conflict. Additionally, if the training set contains a high density of posts where the final sentence is an interrogative prompt (e.g., "What do you think?"), the model has likely learned a strong association between that specific linguistic pattern and the Poll label. The boundary is hard because the intent of a Poll is to gather opinions, which is also the goal of the questions at the end of Opinion and Community posts. Even if your annotations are perfectly consistent, the model is seeing a high degree of overlap in the "instructional" parts of the text.

In order to fix this, we should curate more of these "hard case" examples to expose the boundary between these classes. This involves adding more "Opinion" and "Community" examples that end in questions to the training set, which forces the model to learn that interrogative endings are not exclusive to the "Poll" class. Additionally, contrastive pairs—where posts look similar structurally but differ in content—should be used to train the model to look at the body of the text rather than just the final sentence.


## 4. Sample Classifications

## Correct Classifications
Below are example classifications from the fine-tuned model showcasing successful categorizations across the four labels:

| Post | Predicted Label | Confidence Score | Explanation |
| :--- | :--- | :--- | :--- |
| Taylor Swift’s Courtside Seat from Knicks-Cavaliers Playoff Game Sells at Auction for $7,000 The courtside seat that Taylor Swift sat in at the Cavaliers' arena sold for a hefty price. Memorabilia company The Realest and the Cleveland Cavaliers auctioned off the seat Swift, 36, sat in, as well as the one used by her fiancé Travis Kelce during Game 3 of the Cavaliers' Eastern Conference Finals series against the New York Knicks. According to The Athletic, Swift's seat sold for $7,000 on Sunday, June 14 while Kelce's went for $1,405. The Cavaliers also auctioned off the seats used by by Timothée Chalamet and Kylie Jenner during Game 4. Chalamet's sold for $1,202 and a buyer shelled out $1,505 for Jenner's. The Cavs also sold actor Ben Stiller's Game 4 chair for $732 and rapper Machine Gun Kelly's for $635. | Updates | 0.30 | This prediction is reasonable because the content reports a specific, time-sensitive event regarding Taylor Swift memorabilia, which aligns with the "Updates" category. |
| My Taytoo! I finished this back in January and totally forgot to share it. When I was planning the tattoo I realized almost all my favorite songs involved plants or flowers. All Too Well 🍁 Ivy 🌱 The Lakes 🌸 Don’t Blame Me 🌼 Peter 🕯️ Honey 🐝 and the Folklore cabin represents the best concert of my life - The Eras Tour. I'm beyond happy with how it came out! | Community | 0.27 | This prediction is reasonable because the post focuses on sharing a personal milestone and creative expression with the fan community, which is characteristic of the "Community" category. |
| RANKED: The Tortured Poets Department - The Anthology Album Hey guys!! It's me, u/itookyourmatches, and I'm back again with the annual ranking polls. I'm really excited to see where TTPD has settled with everyone after almost two years, and now how Showgirl fits into our rankings! As always, the link and the rules are below. | Poll | 0.32 | This prediction is reasonable because the post centers on an interactive ranking activity and explicitly invites participation through a link, which serves the primary function of a "Poll." |
| Japanese music magazine INROCK made a special feature about "The 20-Year Journey with Taylor and Our Passionate Feelings" sent in from fans in Japan. | Updates | 0.31 | This prediction is reasonable because the post serves as a factual notice regarding recent media coverage and fan-contributed content, fitting the definition of an "Updates" classification. |

## Incorrect Classifications: Systematic Error Patterns
Below are example classifications from the fine-tuned model showcasing incorrect categorizations across the four labels:

| Post | True Label | Predicted Label | Explanation |
| :--- | :--- | :--- | :--- |
| He Said the way my Blue Eyes Shined: A Masterful Opening Tim McGraw is still an underrated song primarily by virtue of being on Self-Titled. Taylor Swift had these amazing lines to start off the first song of her first album. "He said the way my blue eyes shined Put those Georgia stars to shame that night, I said, 'That's a lie'." This is the opening line to her first song in her discography. I think this series of lines is very strong because it hits like a freight train in terms of imagery. It's a good setup for not just this song, but her debut album and perhaps her entire discography. These lines foreshadow the literary nature of her songwriting. The boyfriend character compared Taylor Swift’s eyes to the Georgia stars, and claims that her eyes would outshine the Georgia stars. Stars are often depicted as beautiful by many writers and artists, so when something is more beautiful than the stars, this is a significant development. To put it another way, her boyfriend is lost in her eyes, and is showing his affection. Stars are also important in navigation, which could be useful for these couples who are lost in each others' eyes. This is very similar to the metaphor Shakespeare uses in Romeo and Juliet, especially with how in love the couples are. The lyricism of these lines are fantastic. The song also ends the same way, which is a very good bookends to bring the listener back to the start and hear it with a new light. This is something she will continue to do in many of her songs. What do you think? | Opinion | Poll | The model misidentified this as a "Poll" because the post concludes with the direct question, "What do you think?" The interrogative ending overrode the extensive, thoughtful literary analysis that comprised the body of the post. |
| The swiftie urge to add a Taylor touch to our hobbies Is it just me or do you also have hobbies that routinely become Taylor-themed? I’ve seen so many swifties be so creative, knitting their own cardigans, baking themed sourdough bread, painting album covers etc.. I love that for us 🫶 Would love to see what everyone is making! | Community | Poll | Despite the post functioning as a celebration of fan creativity, the model flagged it as a "Poll" due to the closing sentence, "Would love to see what everyone is making!" This is a clear case of a conversational flourish being misinterpreted as a structured survey request. |
| I really would love the concept of TS13 to be about luck and flipping the fact 13 is typically an unlucky number. I don’t know whether the title would be something to do with luck or the visuals, colours etc but I just think it’s so cool how 13 is the unlucky number yet Taylor’s favourite and in the case of her story her marriage/true love album will be her thirteenth. I also think it would be fun to just emphasise a lot that this will be the thirteenth one - maybe she could even call it 13 like Adele does with numbers. | Opinion | Poll | The model incorrectly categorized this as a "Poll" because the post's speculative nature and interrogative framing about potential titles and visuals mimicked the structure of a survey, ignoring the core "Opinion" intent of the author. |

## 5. Reflection: Intended vs. Captured Behavior
The core objective in training this model was to build a classifier that identifies the intent of a post: whether it seeks to inform (Updates), foster creative community (Community), analyze works (Opinion), or solicit structured feedback (Poll). However, the resulting model’s decision boundary reveals a significant divergence from these high-level definitions.

The model has effectively learned to map lexical structure rather than semantic intent. Instead of evaluating the "why" behind a post, the model has optimized for a set of high-frequency surface-level markers. Specifically, it has become highly sensitive to:

* Interrogative syntax: Any sequence ending in a question mark is treated as a strong, near-deterministic signal for the "Poll" class.

* Engagement cues: Phrases designed to trigger social interaction (e.g., "What do you think?", "What are your favorites?") serve as overriding weights that collapse the distinction between analytical essays and survey requests.

* Concise reporting markers: The "Updates" label has become tethered to headline-style, declarative sentences, capturing the form of journalism rather than the substance of an update.

The model failed to capture the thematic depth and contextual nuances that distinguish the "Opinion" and "Community" labels. It missed:

* The model cannot differentiate between a question used as a rhetorical closing in an essay (where the answer is secondary to the analysis) and a question that is the actual reason of the post.

* The model effectively ignores the 90% of the post that consists of analysis, creative backstory, or emotional expression if that content is preceded by or ends with a prompt for interaction.

* It treats opinion-sharing posts (whose primary intent is to share such opinion) as a "Poll" simply because the author invites the reader to share a perspective at the very end.


## 6. Spec Reflection
*   **How the spec guided us:** The specification served as a crucial roadmap for defining the hierarchical nature of our label set. By explicitly identifying the four distinct categories—Updates, Community, Opinion, and Poll—the spec provided a clear baseline for annotator instructions and model evaluation. It guided the implementation of a multi-class approach where each post was expected to map to a single primary intent, forcing us to consider the necessity of a "primary intent" hierarchy when building the initial dataset. This structure was instrumental in identifying where our model was failing, as it provided the rigid framework needed to spot when a post’s "primary intent" (Opinion) was being overridden by its "secondary function" (a Poll-like question).
*   **How we diverged:** The implementation diverged from the specification primarily in the assumption that lexical structure would reliably correlate with semantic intent. Our initial spec assumed that clear definitions for "Opinion" and "Poll" would be sufficient for the model to distinguish between an analysis post and a survey. In practice, however, the model ignored our definitions and opted for a "path of least resistance" by overfitting to structural tropes like interrogative endings. We diverged from the original plan because we discovered that the "Poll" category functioned as a powerful, noise-heavy attractor for any post featuring a call to action. We had to pivot from a simple definition-based approach to a more complex, data-centric strategy involving the curation of "hard case" examples to explicitly teach the model the semantic boundaries that our initial label definitions were too simplistic to enforce.

## 7. AI Usage
This project utilized AI assistance to accelerate dataset curation and technical workflows, while maintaining human accountability for all final outputs. For dataset creation, I employed Llama-3.3-70b-versatile (via Groq) to generate a baseline layer of "soft labels" for the 200 collected Reddit posts. I developed a custom Python script to pass post text and my predefined 4-label taxonomy rules to the model. To ensure high data integrity, every AI-suggested label was tracked via an ai_prelabeled metadata column. I conducted a 100% manual audit of the entire dataset, where I reviewed, verified, and—where necessary—overwrote the model's classifications to ensure they strictly adhered to my taxonomy definitions. This process ensured that the final ground truth was defined by human judgment rather than automated tagging.

Beyond data curation, I used **Gemini** as a collaborator for code generation and analysis. I directed Gemini to write scripts for manual labeling audits (`validate_labels.py`) and for extracting specific performance data, such as identifying correctly predicted samples in the test set. I manually reviewed and adjusted the logic of these scripts to ensure they were fit for purpose. Furthermore, I utilized Gemini to generate a script for counting label distribution to ensure balance, and a final script to convert the master dataset into the specific format required for this project, including the fields for text, labels, and notes. In all cases, I served as the final arbiter, reviewing and editing the generated code to ensure it accurately handled the nuances of my dataset structure.