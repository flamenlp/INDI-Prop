# %%
import openai
import langchain
import langchain_openai

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

import time
import pandas as pd

# %%
import os
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model_name="gpt-4o-mini")


# %% [markdown]
# ### FARMER'S PROTEST

# %%
import pandas as pd

data = pd.read_csv('')

data = data[data['Event'] == 'Farmers Protest']


# %%
step_1_template = """Step 1 -- NER
{article}
Identify key entities mentioned in the text to understand the focal points. NOTE: Only include individuals, Organizations/ Groups, Locations, Laws/Policies. Just name the entities, no need to explain.

"""

step_2_template = """Step 2 -- Coreference Resolution
{article}
Now perform Coreference Resolution to resolve pronouns and indirect references to maintain consistency in analysis. Note: Just include the sentences where the entity is coreferred.

"""

step_3_template = """Step 3 -- Entity Relation Analysis
{article}
Now perform relation extraction analysis for the text and identify the relationship between entities and their contexts to detect IF THERE IS potential biases or associations, OR the article has unbiased reporting.

"""

step_4_template = """Step 4 -- Context and Framing Analysis
{article}
Analyze how the article frames entities, events, and legislation based on the analysis from previous steps.

"""

step_5_template = """Step 5 -- Bias Detection
{article}
Based on the analysis done so far, identify whether the article is Neutral, Left-biased, or Right-biased. Do not give explanation just do Classification and Classifiy among -

Bias Classes --

1. Neutral
2. Right-biased
3. Left-biased

**Instructions:**
- Instead of numbering the classes (e.g., 1, 2), provide the full names of the selected Bias class.

**Output Format:**
Bias class: [Bias class Name]

"""

step_6_template = """Step 6 -- Bias Check
{step_5_result}
If the article is 'Neutral', terminate the analysis. If it is 'Left-biased' or 'Right-biased', proceed to Step 7.
"""


step_7_template = """Step 7 -- Narrative Classification
{article}
Since the article is {step_5_result} biased, it is likely to support a {step_5_result} narrative. Based on the analysis so far, write a one-line narrative that best represents the article's perspective.

Narrative Classes --

1. Right Narrative --

    1.1 Glorification of Central Government
    1.2 Vilification of Opposition
    1.3 Framing Anti-Farm Law Protests as Subversive
    1.4 Justifying Farm Laws by Critiquing Current Policies
    1.5 Criticising Global Figures and Celebrities


2. Left Narrative --

    2.1 Vilification of Central Government
    2.2 Depicting Farmers as Victims
    2.3 Emphasizing Global and Celebrity Endorsements
    2.4 Accusing Media and Government of Manipulation

**Instructions:**
- Instead of numbering the classes (e.g., 2.1, 2.2), provide the full names of the selected narratives.
- Only include the **subclasses** of Left, Right, and Both-Side Narratives in the output, not the main categories themselves.

**Output Format:**
Narrative Classes: [list of narrative names separated by commas]
Brief Reasoning: [short explanation of why they were classified as such]

"""

def analyze_article(article):

    step_1_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article"], template=step_1_template))
    step_1_result = step_1_chain.run(article=article)

    step_2_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1"], template=step_2_template))
    step_2_result = step_2_chain.run(article=article, step_1=step_1_result)

    step_3_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2"], template=step_3_template))
    step_3_result = step_3_chain.run(article=article, step_1=step_1_result, step_2=step_2_result)

    step_4_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3"], template=step_4_template))
    step_4_result = step_4_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result)

    step_5_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3", "step_4"], template=step_5_template))
    step_5_result = step_5_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result, step_4=step_4_result)

    step_6_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["step_5_result"], template=step_6_template))
    step_6_result = step_6_chain.run(step_5_result=step_5_result)

    bias_detected = step_5_result.strip()

    if "Neutral" in bias_detected:
        return bias_detected, None, None

    step_7_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3", "step_4", "step_5_result"], template=step_7_template))
    step_7_result = step_7_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result, step_4=step_4_result, step_5_result=step_5_result)

    if "Brief Reasoning:" in step_7_result:
        narrative_class = step_7_result.split("Brief Reasoning:")[0].replace("Narrative Classes:", "").strip()
        reasoning = step_7_result.split("Brief Reasoning:")[1].strip()
    else:
        narrative_class = step_7_result
        reasoning = ""
    
    return bias_detected, narrative_class, reasoning

detected_biases = []
narrative_classes = []
reasonings = []

for index, row in data.iterrows():
    article = row['Article']  
    bias, narrative_class, reasoning = analyze_article(article)

    detected_biases.append(bias if bias else "")
    narrative_classes.append(narrative_class if narrative_class else "")
    reasonings.append(reasoning if reasoning else "")
    
    print(f"{index+1} Articles Processed - Bias: {bias}")

    if (index + 1) % 5 == 0:
        print(f"Processed {index + 1} articles. Taking a short break...")
        time.sleep(10)  

data.loc[data.index, 'Detected Bias'] = detected_biases
data.loc[data.index, 'Narrative Classes Framework'] = narrative_classes
data.loc[data.index, 'Reasoning Framework'] = reasonings

data.to_csv("Result_Farmers.csv", index=False)


# %% [markdown]
# ### CAA

# %%
data = data[data['Event'] == 'CAA']

# %%
step_1_template = """Step 1 -- NER
{article}
Identify key entities mentioned in the text to understand the focal points. NOTE: Only include individuals, Organizations/ Groups, Locations, Laws/Policies. Just name the entities, no need to explain.

"""

step_2_template = """Step 2 -- Coreference Resolution
{article}
Now perform Coreference Resolution to resolve pronouns and indirect references to maintain consistency in analysis. Note: Just include the sentences where the entity is coreferred.

"""

step_3_template = """Step 3 -- Entity Relation Analysis
{article}
Now perform relation extraction analysis for the text and identify the relationship between entities and their contexts to detect IF THERE IS potential biases or associations, OR the article has unbiased reporting.

"""

step_4_template = """Step 4 -- Context and Framing Analysis
{article}
Analyze how the article frames entities, events, and legislation based on the analysis from previous steps.

"""

step_5_template = """Step 5 -- Bias Detection
{article}
Based on the analysis done so far, identify whether the article is Neutral, Left-biased, or Right-biased. Do not give explanation just do Classification and Classifiy among -

Bias Classes --

1. Neutral
2. Right-biased
3. Left-biased

**Instructions:**
- Instead of numbering the classes (e.g., 1, 2), provide the full names of the selected Bias class.

**Output Format:**
Bias class: [Bias class Name]

"""

step_6_template = """Step 6 -- Bias Check
{step_5_result}
If the article is 'Neutral', terminate the analysis. If it is 'Left-biased' or 'Right-biased', proceed to Step 7.
"""

step_7_template = """Step 7 -- Narrative Classification
{article}
Since the article is {step_5_result} biased, it is likely to support a {step_5_result} narrative. Based on the analysis so far, write a one-line narrative that best represents the article's perspective.

Narrative Classes --

1. Right Narrative --

        1.1 Glorification of CAA
        1.2 Vilification of the Center’s Opposition
        1.3 Opposition spreading misinformation and fear.
        1.4 Framing Anti-CAA Protests as Subversive, Anti-Hindu, or Misguided
        1.5 Anti-CAA Protests Are a Pre-Planned and Funded Conspiracy.
        1.6 Delegitimization of Critics
        1.7 Potraying central goverment as the Champion of National Interests

2. Left Narrative --
        2.1 Vilification of CAA
        2.2 Vilification of Central goverment
        2.3 Glorifying Anti-CAA Protesters.
        2.4 Framing Anti-CAA Protesters as Victims.

**Instructions:**
- Instead of numbering the classes (e.g., 2.1, 2.2), provide the full names of the selected narratives.
- Only include the **subclasses** of Left, Right, and Both-Side Narratives in the output, not the main categories themselves.

**Output Format:**
Narrative Classes: [list of narrative names separated by commas]
Brief Reasoning: [short explanation of why they were classified as such]

"""

def analyze_article(article):
    
    step_1_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article"], template=step_1_template))
    step_1_result = step_1_chain.run(article=article)

    step_2_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1"], template=step_2_template))
    step_2_result = step_2_chain.run(article=article, step_1=step_1_result)

    step_3_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2"], template=step_3_template))
    step_3_result = step_3_chain.run(article=article, step_1=step_1_result, step_2=step_2_result)

    step_4_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3"], template=step_4_template))
    step_4_result = step_4_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result)

    step_5_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3", "step_4"], template=step_5_template))
    step_5_result = step_5_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result, step_4=step_4_result)

    step_6_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["step_5_result"], template=step_6_template))
    step_6_result = step_6_chain.run(step_5_result=step_5_result)

    bias_detected = step_5_result.strip()

    if "Neutral" in bias_detected:
        return bias_detected, None, None

    step_7_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["article", "step_1", "step_2", "step_3", "step_4", "step_5_result"], template=step_7_template))
    step_7_result = step_7_chain.run(article=article, step_1=step_1_result, step_2=step_2_result, step_3=step_3_result, step_4=step_4_result, step_5_result=step_5_result)

    if "Brief Reasoning:" in step_7_result:
        narrative_class = step_7_result.split("Brief Reasoning:")[0].replace("Narrative Classes:", "").strip()
        reasoning = step_7_result.split("Brief Reasoning:")[1].strip()
    else:
        narrative_class = step_7_result
        reasoning = ""
    
    return bias_detected, narrative_class, reasoning

detected_biases = []
narrative_classes = []
reasonings = []

for index, row in data.iterrows():
    article = row['Article']  
    bias, narrative_class, reasoning = analyze_article(article)

    detected_biases.append(bias if bias else "")
    narrative_classes.append(narrative_class if narrative_class else "")
    reasonings.append(reasoning if reasoning else "")
    
    print(f"{index+1} Articles Processed - Bias: {bias}")

    if (index + 1) % 5 == 0:
        print(f"Processed {index + 1} articles. Taking a short break...")
        time.sleep(10)  

data.loc[data.index, 'Detected Bias'] = detected_biases
data.loc[data.index, 'Narrative Classes Framework'] = narrative_classes
data.loc[data.index, 'Reasoning Framework'] = reasonings

data.to_csv("result_CAA.csv", index=False)


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



