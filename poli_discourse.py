!pip install colab-xterm
%load_ext colabxterm

!curl https://ollama.ai/install.sh | sh

%xterm
#ollama serve

!ollama pull qwen2.5:7b 
#can also use Deepseek-r1: 8b, Llama 3.2:1b, llama 3.2:1b, Gemma 3:4b, Mistral:7b, Phi-4:14b

!pip install langchain_ollama

import pandas as pd
import time


# Load the test dataset
df = pd.read_excel('/content/Prompt_dataset.xlsx')

# Install necessary libraries if not already installed
# pip install langchain-core langchain-ollama pandas openpyxl

import re
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# LLM
llm = ChatOllama(temperature=0, model="llama3.2")

# Prompt Template
tagging_prompt = ChatPromptTemplate.from_template(
    """
You are a social media analyst tasked with classifying the Indian political leaning of Indian social media posts.
Classify the following text into one of the following categories:

- 'Right wing': associated with nationalist, conservative, extremist, religious, orthodox or pro-establishment viewpoints. the text may contain criticism of Congress/Rahul Gandhi. the text might be on pro bjp or pro narendra modi. the key words may include hindu, islam, ram-mandir, muslim, modiji
- 'Left wing': associated with liberal, progressive or anti-establishment views. it may contain text in support of congress or AAP, criticism of India, the goverment, bjp or rss. And the key words may include lobby, dictatorship, propaganda, saffron, bhakt and other mocking wordplay targetted at indians
- 'Central wing': neutral, balanced, or moderate views that do not lean strongly Right wing or Left wing, and may contain statements, news report like text, personal views, commentary on economy, commentary on any political event and commentary on what is left wing, right wing and central wing, balanced criticisms.


+ Examples for Right wing (Hindi few shot):
  - Text: " सावधान एकनाथ शिंदे जी का ऐलान गौ हत्यारे और धर्मान्तरण वाले मौलाना कान खोलकर सुन लें महाराष्ट्र में धर्मान्तरण अब पूरी तरह बंद करने की घोषणा हो गयी है" → Right wing
  - Text: "अमेरिका के पूर्व राष्ट्रपति डोनाल्ड ने पहले से ही कहा है कि जहां इस्लामिक कट्टर पंथी रहेंगे वहां आतंकवाद होना तय है" → Right wing

+ Examples for Left wing (Hindi few shot):
  - Text: "भारत में इतने तेजस्वी नेताओं ने  18 , 18 काम करने के बाद भारत में  ग़रीबी, बेरोजगारी, भुखमरी,घोटाले, महगाई, रुपये के घटती वैल्यू,15 करोड़ का  जुमला, किसानों की आय दोगनी करने का जुमला। भारत को सोने की चिड़ियाँ बनाने का जुमला,🚩🇮🇳🖋 जय हिंद जय भारत" → Left wing
  - Text: "ना कोर्ट ना कचहरी सिर्फ #जंगलराज ? #तानाशाही ?" → Left wing

+ Examples for Right wing (English few shot):
  - Text: "modi had promised congress free hindustan was holding government states three states voted opposition mere lies now penance they had doubted honest man will mandate modiji heavily wash off thier sin had not modi met this promise" → Right wing
  - Text: "rahul isnt mature politician rahul managing director from congress companies har har modi ghar ghar modi modi jeise imandar milna moskil haiji modi the great leader the world support modi" → Right wing
  - Text: "rahulgandhi must thank modi ji for this change in kashmir otherwise during congress it was worse than hell" → Right wing

+Examples for Left wing (English few shot):
  - Text: "Modi was supposed to pass a strong Lokayukta Bill as per his campaign promises in 2014. Its been 7 years, but he is busy fulfilling his RSS Hindutva promises." → Left wing
  - Text: "What a rubbish argument about constitution! 90% Indians knows nothing about a mumbo-jumbo called constitution. It was a bloody greed of Rs, 8,500 which had caused a upset for BJP, there was no other reason behind it." → Left wing
  - Text: "Because that's what it has come up to. I wish more news like these gets published and reduce the religious hatred. In pre modi era, people were talking about economy, corruption, jobs etc. Now everyone talks about cow urine, beef, Pakistan and other utter nonsense." → Left wing

+Example for Central wing (English few shot):
  - Text: "maharashtra bjp leader demands renaming of ahmednagar after maratha queen ahilyabaiholkar ahilyanagar ahmednagar eknathshinde devendrafadnavis" → Central wing
  - Text: "manishsisodiaarrested aap workers march towards the bjp headquarter in jammu raises slogans against centre aap bjp jammu manishsisodiaarrested aapexposed manishsisodia arvindkejriwal watch" → Central wing
  - Text: "first time voters ready to cast their votes at don bosco higher secondary school ii polling station under kohima dist nagaland wethenagas dimapur kohima ndpp nagalandcongress jdu news ljp nagalandelection2Central wing23 nagalandelection bjp npf nagalandnews" → Central wing

+ **Guard‑rails:**
  - If the input reads like a news headline, article, tweet, or report that includes actions/events/statements without opinion, emotional language, or ideological framing, label it Central wing and skip further analysis.
  - Examples of such texts include announcements, arrests, marches, slogans raised, events involving political figures/parties—unless there's clear ideological framing (e.g., calling someone corrupt, fascist, etc.).
  - Treat actions/events alone (e.g., "X marched", "Y protested", "Z arrested") as Central wing unless interpretation or bias is introduced.
  - Do not assume ideology based on parties mentioned alone—focus on tone and content, not just entity names.


Respond with only the category name, without any additional text.

{input}
"""
)


# Function to clean LLM output by removing <think>...</think> if exists
def erase_think_section(text):
    """
    Removes everything up to and including the closing </think> tag.
    """
    pattern = r"^[\s\S]*</think>(?![\s\S]*</think>)"
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return cleaned_text.strip()

def process_and_append(df, batch_size=25, text_column='Text', output_column='PredictedWing'):
    """
    Processes the DataFrame in batches, classifies each 'text_column' entry,
    appends the result into 'output_column', and returns the updated DataFrame.
    """
    predictions = []
    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        for _, row in batch.iterrows():
            input_text = row[text_column]
            prompt = tagging_prompt.invoke({'input': input_text})
            response = llm.invoke(prompt)

            # Apply erase_think_section to clean the output
            cleaned_output = erase_think_section(response.content)
            predictions.append(cleaned_output)

    # Append predictions to a new column
    df[output_column] = predictions
    return df

# Load your dataframe
# df = pd.read_excel('your_file.xlsx')  # <--- Make sure you load it first

# Run the function
df_with_predictions = process_and_append(df, batch_size=25)

# Save back to Excel
df_with_predictions.to_excel('original_data_with_predictions.xlsx', index=False)
print("✅ Saved predictions in 'PredictedWing' column to original_data_with_predictions.xlsx")


import pandas as pd

# 1. Load your labeled dataset (with ground-truth 'Label') and predictions
df = pd.read_excel('original_data_with_predictions.xlsx')

# 2. Ensure the necessary columns exist
label_col     = 'Label'
predicted_col = 'PredictedWing'

if label_col in df.columns and predicted_col in df.columns:
    # 3. Create a boolean “is_correct” column
    df['is_correct'] = df[label_col] == df[predicted_col]

    # 4. Compute raw counts
    total_examples   = len(df)
    correct_count    = df['is_correct'].sum()
    incorrect_count  = total_examples - correct_count

    # 5. Compute accuracy
    accuracy         = correct_count / total_examples

    # 6. Print out results
    print("RIGHT WING")
    print(f"Total examples   : {total_examples}")
    print(f"Correctly labeled: {correct_count}")
    print(f"Incorrect labels : {incorrect_count}")
    print(f"Accuracy         : {accuracy:.2%}")
else:
    print(f"One or both required columns ('{label_col}', '{predicted_col}') are missing.")


import pandas as pd

# 1. Load your labeled dataset (with ground-truth 'Label') and predictions
df = pd.read_excel('original_data_with_predictions.xlsx')

# 2. Ensure the necessary columns exist
label_col     = 'Label'
predicted_col = 'PredictedWing'

if label_col in df.columns and predicted_col in df.columns:
    # 3. Create a boolean “is_correct” column
    df['is_correct'] = df[label_col] == df[predicted_col]

    # 4. Compute raw counts
    total_examples   = len(df)
    correct_count    = df['is_correct'].sum()
    incorrect_count  = total_examples - correct_count

    # 5. Compute accuracy
    accuracy         = correct_count / total_examples

    # 6. Print out results
    print("LEFT WING")
    print(f"Total examples   : {total_examples}")
    print(f"Correctly labeled: {correct_count}")
    print(f"Incorrect labels : {incorrect_count}")
    print(f"Accuracy         : {accuracy:.2%}")
else:
    print(f"One or both required columns ('{label_col}', '{predicted_col}') are missing.")


import pandas as pd

# 1. Load your labeled dataset (with ground-truth 'Label') and predictions
df = pd.read_excel('original_data_with_predictions.xlsx')

# 2. Ensure the necessary columns exist
label_col     = 'label'
predicted_col = 'PredictedWing'

if label_col in df.columns and predicted_col in df.columns:
    # 3. Create a boolean “is_correct” column
    df['is_correct'] = df[label_col] == df[predicted_col]

    # 4. Compute raw counts
    total_examples   = len(df)
    correct_count    = df['is_correct'].sum()
    incorrect_count  = total_examples - correct_count

    # 5. Compute accuracy
    accuracy         = correct_count / total_examples

    # 6. Print out results
    print("CENTRAL WING")
    print(f"Total examples   : {total_examples}")
    print(f"Correctly labeled: {correct_count}")
    print(f"Incorrect labels : {incorrect_count}")
    print(f"Accuracy         : {accuracy:.2%}")
else:
    print(f"One or both required columns ('{label_col}', '{predicted_col}') are missing.")


#for single shot testing, where one input is manually given at a time, and recieve the labelling with chain-of-thought

inp = "it was year 2Central wingRight wingCentral wing when amit shah ji was home minister of gujarat and p chidambaram was then finance minister it was indian national congress who continuously raided opposition specially bjp by wrongly using the government agencies and amit shah was jailed in corruption cases"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

response.content
