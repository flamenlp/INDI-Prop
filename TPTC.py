# %%
import openai
import langchain
import langchain_openai

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

# %%
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

llm = ChatOpenAI(model_name="gpt-4o-mini")

# %%
import pandas as pd

df_gt = pd.read_csv("")

df = df_gt

# %%
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(
    model="gpt-4o-mini",   
    temperature=0,
    max_tokens=50
)

prompt_template = PromptTemplate(
    input_variables=["article"],
    template="""
    You are a classification expert for Propaganda. You have to classify the given article into one or more of the following classes. Each class has a definition given.

'G1': Emotional manipulation → Statements that rely heavily on triggering emotional responses rather than presenting factual or logical arguments.  

'G2': Distraction and Misdirection → Statements that aim to divert attention away from the central issue or argument by introducing irrelevant, distorted, or ambiguous information.  

'G3': Simplistic thinking → Statements that simplify complex issues into overly simplistic narratives, either by reducing causes, forcing binary choices, or using authoritative statements without evidence.  

'G4': Attack and Defamation → Statements that undermine credibility by targeting individuals/groups with negative portrayals.  

'G5': Manipulation through Popularity and Authority → Persuasion relying on expert credibility or peer pressure.  

'G6': Misrepresentation and Distortion → Altering perception of reality through straw man, exaggeration, minimization.  

'G7': Undermining Trust → Introducing uncertainty, confusion, skepticism, eroding confidence in claims or sources.  

'G0': No Narrative.  

---

Read the article below:

Article: {article} 

---

⚠️ IMPORTANT:  
write only class name from the classes: [G0,G1,G2,G3,G4,G5,G6,G7] 

Return the output strictly in the format given below.:
{{
    [predicted class 1, predicted class 2, ...]
}}

"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# %%
import pandas as pd
import json
from tqdm import tqdm

def query_gpt(article):
    response = chain.run(article=article)

    try:
        parsed = json.loads(response)
        spans = {cls: ", ".join(vals) if isinstance(vals, list) else vals 
                 for cls, vals in parsed.items()}
    except Exception:
        spans = {"raw_response": response}
    return spans

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Articles"):
    article = row["Article"]
    Ground_Truth = row["Group_Labels"]

    spans = query_gpt(article)

    results.append({
        "Article": article,
        "Ground_Truth": Ground_Truth,
        "Predicted": json.dumps(spans, ensure_ascii=False)  
    })

results_df = pd.DataFrame(results)




# %%


# %%
import pandas as pd
import re

def clean_pred(val):
    if pd.isna(val):
        return ""
    groups = re.findall(r"G\d+", str(val))
    return ",".join(groups)

results_df["Predicted_clean"] = results_df["Predicted"].apply(clean_pred)


# %%
results_df = results_df.drop(columns=["Predicted"])

results_df = results_df.rename(columns={"Predicted_clean": "Predicted"})


# %%
import pandas as pd

df = results_df

df['Ground_Truth'] = df['Ground_Truth'].str.strip().replace("No Technique", "G0")

print(df.head())


# %%
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(
    model="gpt-4o-mini",   
    temperature=0,
    max_tokens=1200
)

prompt_template = PromptTemplate(
    input_variables=["article", "prediction"],
    template="""
Read the article below:

Article: {article} 

My finetuned model predicted these classes: {prediction}.  
For all the classes present in this prediction, their definitions are given below. Read carefully.

'G1': Emotional manipulation → Statements that rely heavily on triggering emotional responses rather than presenting factual or logical arguments.  
Task: Extract the exact spans. ONLY exact spans.

'G2': Distraction and Misdirection → Statements that aim to divert attention away from the central issue or argument by introducing irrelevant, distorted, or ambiguous information.  
Task: Extract the exact spans. ONLY exact spans.

'G3': Simplistic thinking → Statements that simplify complex issues into overly simplistic narratives, either by reducing causes, forcing binary choices, or using authoritative statements without evidence.  
Task: Extract exact spans. ONLY exact spans.

'G4': Attack and Defamation → Statements that undermine credibility by targeting individuals/groups with negative portrayals.  
Task: Extract exact spans. ONLY exact spans.

'G5': Manipulation through Popularity and Authority → Persuasion relying on expert credibility or peer pressure.  
Task: Extract exact spans. ONLY exact spans.

'G6': Misrepresentation and Distortion → Altering perception of reality through strawman, exaggeration, minimization.  
Task: Extract exact spans. ONLY exact spans.

'G7': Undermining Trust → Introducing uncertainty, confusion, skepticism, eroding confidence in claims or sources.  
Task: Extract exact spans. ONLY exact spans.

---

⚠️ IMPORTANT:  
Now extract spans ONLY for the following predicted classes: {prediction}  

Return the output strictly in the format given below.:
{{
  "G1": [span1: "span1", span2: "span2"],
  "G2": [],
  "G6": [span1: "span1"]
}}

"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)


# %%
#### COARSE GRAIN START

# %%
import pandas as pd
import json
from tqdm import tqdm

def query_gpt(article, prediction):
    response = chain.run(article=article, prediction=prediction)
    try:
        parsed = json.loads(response)
        spans = {cls: ", ".join(vals) if isinstance(vals, list) else vals 
                 for cls, vals in parsed.items()}
    except Exception:
        spans = {"raw_response": response}
    return spans

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Articles"):
    article = row["Article"]
    prediction = row["Predicted"]

    spans = query_gpt(article, prediction)

    results.append({
        "Article": article,
        "Predicted": prediction,
        "Spans": json.dumps(spans, ensure_ascii=False)  
    })

results_df = pd.DataFrame(results)



# %%


# %%
import pandas as pd
import json
import re

df_out = results_df

processed_spans = []

for raw_json in df_out["Spans"]:
    try:        
        outer = json.loads(raw_json)
        raw = outer.get("raw_response", "{}")

        fixed = re.sub(r'span\d+:\s*', '', raw)

        parsed = json.loads(fixed)

        all_spans = []
        for spans in parsed.values():
            if spans:
                all_spans.extend(spans)

        processed_spans.append(" | ".join(all_spans))
    except Exception:
        processed_spans.append("")

df_out["processed_spans"] = processed_spans

# %%


# %%


# %%


# %%
df_out['processed_spans'] = df_out['processed_spans'].fillna('No Tags')

print(df_out['processed_spans'].tail())

# %%
coarse_to_fine = {
    "Causal Oversimplification": ["G3"],
    "Black-and-white Fallacy": ["G3"],
    "Straw Man": ["G2", "G6"],
    "Whataboutism": ["G2"],
    "Reductio ad Hitlerum": ["G4"],
    "Red Herring": ["G2"],
    "Loaded Language": ["G1"],
    "Name Calling and Labeling": ["G1", "G4"],
    "Appeal to emotion/fear/prejudice": ["G1"],
    "Slogans": ["G1"],
    "Flag-Waving": ["G1"],
    "Exaggeration and Minimization": ["G6"],
    "Thought-terminating Cliches": ["G3"],
    "Bandwagon": ["G5"],
    "Smears": ["G4"],
    "Obfuscation/ intentional vagueness/ confusion": ["G2", "G7"],
    "Doubt": ["G7"],
    "Appeal to Authority": ["G5"],
    "Assertion": ["G3"],
    "Glittering Generalities (Virtue)": ["G1"],
    "No Technique": ["G0"]
}


# %%
from collections import defaultdict

fine_map = defaultdict(list)
for fine, groups in coarse_to_fine.items():
    for g in groups:
        fine_map[g].append(fine)


# %%
df = df_out

# %%
fine_preds = []

for _, row in df.iterrows():
    val = row["Predicted"]

    if pd.isna(val):
        fine_preds.append("")  
        continue

    coarse_preds = [c.strip() for c in str(val).split(",") if c.strip()]

    fine_list = []
    for c in coarse_preds:
        if c in fine_map:  
            fine_list.extend(fine_map[c])

    fine_preds.append(", ".join(sorted(set(fine_list))))

df["Fine_Grain"] = fine_preds



# %%
##### FINE GRAIN START

# %%
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def build_finegrain_prompt(article, spans, fine_grain_list):
    return f"""
You are an expert annotator.

Article: {article}

From the list {fine_grain_list}, determine which fine-grain propaganda techniques are present in the spans below and assign spans accordingly.

Spans: {spans}

Instructions:
- Match each span with the most appropriate fine-grain technique(s). All spans must be matched only with techniques from the list: {fine_grain_list}. (NOTE: Don't assign NaN to any span)
- If a span clearly matches multiple techniques, assign it to all relevant ones.   
- Output strictly in the following format (one technique per line):  
- Only if list have 'No Technique', then output: NaN

Technique: ['span1', 'span2']
Technique: ['span3']

Example : 

List of Fine-Grain Techniques: Assertion,Name Calling and Labeling,Slogans,Causal Oversimplification,Black-and-white Fallacy

Spans: anti-CAA jihadi mob. | “Nara e Taqbeer, Allahu Akbar” | Muslim anti-CAA rioters go on a rampage in Delhi against the Citizenship Law, seeking to draw international attention amidst the high-profile visit of US President Donald Trump. 

OutPut: Name Calling and Labeling: ['anti-CAA jihadi mob.']
        Slogans: ['“Nara e Taqbeer, Allahu Akbar”']
        Assertion: ['Muslim anti-CAA rioters go on a rampage in Delhi against the Citizenship Law, seeking to draw international attention amidst the high-profile visit of US President Donald Trump.']"

"""

prompt_template = PromptTemplate(
    input_variables=["article", "spans", "fine_grain_list"],
    template=build_finegrain_prompt("{article}", "{spans}", "{fine_grain_list}")
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def query_finegrain(article, spans, fine_grain_list):
    response = chain.run(article=article, spans=spans, fine_grain_list=fine_grain_list)
    return response

finegrain_results = []
for _, row in df.iterrows():
    article = row["Article"]
    spans = row["processed_spans"]
    fine_grain_list = row["Fine_Grain"]  # 👈 take list from CSV column
    print(f"Processing article with fine-grain list: {fine_grain_list}")
    response = query_finegrain(article, spans, fine_grain_list)
    finegrain_results.append(response)

df["Fine_Grain_Spans"] = finegrain_results


# %%
###--------FINE GRAIN END---

# %%
technique_to_tag = {
    "Causal Oversimplification": "T1",
    "Black-and-white Fallacy": "T2",
    "Straw Man": "T3",
    "Whataboutism": "T4",
    "Reductio ad Hitlerum": "T5",
    "Red Herring": "T6",
    "Loaded Language": "T7",
    "Name Calling and Labeling": "T8",
    "Appeal to emotion/fear/prejudice": "T9",
    "Slogans": "T10",
    "Flag-Waving": "T11",
    "Exaggeration and Minimization": "T12",
    "Thought-terminating Cliches": "T13",
    "Bandwagon": "T14",
    "Smears": "T15",
    "Obfuscation/ intentional vagueness/ confusion": "T16",
    "Doubt": "T17",
    "Appeal to Authority": "T18",
    "Assertion": "T19",
    "Glittering Generalities (Virtue)": "T20"
}


# %%
import pandas as pd

def replace_with_tags(text):
    if pd.isna(text):
        return text
    for tech, tag in technique_to_tag.items():
        text = text.replace(tech, tag)
    return text

df["Fine_Grain_Spans_Tagged"] = df["Fine_Grain_Spans"].apply(replace_with_tags)

# %%
df["Fine_Grain_Spans_Tagged"] = df["Fine_Grain_Spans_Tagged"].fillna("No Tags")

# %%
df.head()


