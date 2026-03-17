1. Data Sources 
• Kaggle: Publicly available data sets on online discourse.
2. Collection Process 
• Extracted ~500 tweets via existing Kaggle datasets. 
• Extracted ~500 reddit comments via existing Kaggle datasets. 
• Ensured diversity of language use (English, Hindi, code-mixed). 
• Filtered out discourse related to Indian politics using NLTK and a curated list of 
keywords. 
3. Manual Annotation 
• Categories: Right wing, Left wing, Central wing. 
• Guidelines: 
o Right wing: nationalist, conservative, orthodox; key terms Hindu, ram-mandir, 
Muslim… 
o Left wing: liberal, anti-establishment, communist; keywords dictatorship, 
propaganda… 
o Central wing: neutral, balanced; news-style or mixed commentary. 
• Two-step process: initial label by me, review by supervisor; discrepancies resolved 
via discussion.

4. Development Platform 
• Google Colab Free Tier: primary coding environment, limited to ~15 GB GPU 
memory. 
• Local Workstation: occasional preprocessing in Anaconda Jupyter Python notebook. 
5. Platform and Environment Setup 
➢ The project was implemented in Google Colab, a cloud-based Jupyter notebook 
environment that allows access to free GPUs and seamless integration with Python 
libraries.
➢ Ollama, a lightweight local inference engine for running large language models
6. Language Model Deployment (LLM) 
➢ The open-source llm model was pulled using Ollama
➢ Langchain integration was used for interfacing with the Mistral:7b model
7. Data Handling 
➢ The dataset was loaded from Excel and predictions were saved back using pandas
8. Libraries and Frameworks 
• Pandas: data manipulation and analysis. 
• LangChain_Ollama: for prompt templating and model calls.


Prompt Structure and Purpose: 
1. Role Definition & Task Clarity 
• Objective: Frame the LLM’s role and classification goal 
• Method: 
o Explicitly defines the task:  
"You are a social media analyst tasked with 
classifying Indian political leaning." 
o Sets boundaries to avoid ambiguity, ensuring the model focuses solely on 
political classification. 
2. Category Definitions with Linguistic & Contextual Cues 
• Right-wing: 
o Keywords: Hindu, Islam, Ram Mandir, Modi (pro-BJP/Narendra Modi 
rhetoric). 
o Tone: Nationalist, conservative, or critical of opposition (e.g., Congress/Rahul 
Gandhi).
• Left-wing: 
o Keywords: Dictatorship, propaganda, saffron, bhakt (anti-BJP/RSS rhetoric). 
o Tone: Critical of government, progressive, or supportive of Congress.
• Central-wing: 
o Criteria: Neutral reports, economic commentary, or balanced critiques. 
o Guardrail: Default category for factual/event-based texts (e.g., news 
headlines, protests) unless ideological framing is explicit.
3. Multilingual Adaptation 
• Hindi-English Code-Mixing: 
o Includes Hindi few-shot examples  to train the model on code-switched texts.
4. Structured Few-Shot Learning 
• Examples: 
o Provides annotated samples (In Hindi and English) to demonstrate: 
▪ Right-wing: Pro-Hindu narratives, Modi-centric praise. 
▪ Left-wing: Anti-government critiques, sarcasm. 
o Balances language diversity and ideological patterns. 
• Impact: Teaches the model to map subtle linguistic patterns (e.g., wordplay, tone) to 
categories.
5. Guardrails for Neutrality & Bias Mitigation 
• Key Rules: 
- Central-wing Default: Forces neutrality for event reports lacking 
opinion/emotion.
- No Entity-Based Assumptions: Prohibits labelling based solely on party names 
(e.g., mentioning BJP ≠ Right-wing). 
- Focus on Tone/Content: Prioritizes linguistic cues (e.g., mocking terms 
like bhakt) over named entities. 
• Purpose: Reduces overfitting to keywords and ensures ideological framing drives 
classification.


Summary 
The prompt combines explicit definitions, culturally contextual examples, and bias-mitigation 
rules to guide the LLM. By focusing on linguistic cues (keywords, tone) over named entities 
and enforcing neutrality defaults, the approach balances accuracy with contextual fairness. The 
inclusion of Hindi-English few-shot learning addresses India’s multilingual social media 
landscape, while iterative testing (constrained by GPU limits) validated the llm’s efficiency for 
this task. This structured prompt engineering framework ensures replicability for similar 
sociopolitical NLP applications.
