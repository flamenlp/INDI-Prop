# INDI-PROP

This repository contains the INDI-PROP dataset along with the implementation of the FANTA & TPTC frameworks, as introduced in our paper:

📄 Paper: [*Fine-grained Narrative Classification in Biased News Articles*](https://lrec.elra.info/lrec2026-main-665?shem=rimspwouoe,)  
📅 Presented at: Language Resources and Evaluation Conference (LREC), 2026  
📍 Palma, Mallorca (Spain) | 🗓 11–16 May 2026

## Overview

INDI-PROP is an ideologically grounded dataset for studying bias, narrative framing, and persuasive techniques in Indian news media. The resource is built around two major socio-political events:

1. CAA/NRC

2. Farmers' Protest 

Each article is annotated at multiple levels to capture:

1. Article-level bias

2. Fine-grained narratives

3. Persuasive techniques 

This enables analysis of how ideological framing and persuasion co-exist in media discourse.

## Dataset Statistics

| Component | Count |
|----------|------|
| Total Articles | 1,266 |
| Events Covered | 2 |
| Bias Labels | 3 (Pro-Gov, Pro-Opp, Neutral) |
| Fine-Grain Narrative Categories | 11 (CAA/NRC), 9 (Farmer's Protest) |
| Fine-Grain Persuasion Techniques Categories | 20 |

## Framework

We introduce two multi-hop reasoning frameworks to enable structured ideological analysis.

<p align="center">
  <img src="Images/framework.png" width="900"/>
</p>

FANTA performs article bias classification and narrative detection by modeling entities, relations, and contextual framing. And TPTC identifies persuasive strategies using a two-stage pipeline — first detecting conceptual persuasion categories, and then mapping them to fine-grained techniques.

## Dataset Access

The dataset is available for research purposes only.

To request access, please fill out the form below:

👉 [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSegB-fPJ7H4Wsn2nGWzty2Ju5RcV8wDT9uYmjfvt6FADC4CcA/viewform?usp=dialog)

## Usage Policy

By requesting access, you agree to:

- Use the dataset strictly for academic or research purposes.

- Do not use it for political targeting, profiling, or manipulation.

- Cite the original paper in any published work.

The data reflects real-world political discourse and may contain ideological or sensitive content.
Users are encouraged to interpret and use it responsibly.

👉 The dataset is released to support transparency and computational media analysis.

## Citation

If you use this dataset or framework, please cite:

```bibtex
@inproceedings{afroz-etal-2026-fine,
  title = {Fine-grained Narrative Classification in Biased News Articles},
  author = {Afroz, Zeba and Vardhan, Harsh and Bhakuni, Pawan and Punia, Aanchal and Kumar, Rajdeep and Akhtar, Md. Shad},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {8431--8445},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/2ddvvr4zijyh},
  abstract = {Narratives are the cognitive and emotional scaffolds of propaganda. They organize isolated persuasive techniques into coherent stories that justify actions, attribute blame, and evoke identification with ideological camps. In this paper, we propose a novel fine-grained narrative classification in biased news article. We also explore article-bias classification as the pre-cursor task to narrative classification and fine-grained persuassive technique identification. We develop INDI-PROP, the first ideologically grounded fine-grain narrative dataset with multi-level annotation for analyzing propaganda in Indian news media. Our dataset INDI-PROP comprises 1,266 articles focusing on two polarizing socio-political events in recent times: CAA/NRC and the Farmers’ protest. Each article is annotated at three hierarchical levels: (i) ideological article-bias (pro-government, pro-opposition, neutral), (ii) event-specific fine-grained narrative frames anchored in ideological polarity and communicative intent, and (iii) persuasive techniques. We propose FANTA and TPTC, two GPT-4o guided multi-hop prompt-based reasoning frameworks for the bias, narrative, and persuasive technique classification. FANTA leverages multi-layered communicative phenomenon by integrating information extraction and contextual framing for hierarchical reasoning. On the other hand, TPTC adopts systematic decomposition of persuasive cues via a two-stage approach. Our evaluation suggest substantial improvement over underlying baselines in each case.}
}
```

## Contact

For questions or collaborations:

- **Zeba Afroz** — [zebaa@iiitd.ac.in](mailto:zebaa@iiitd.ac.in)
- **Harsh Vardhan** — [harsh25001@iiitd.ac.in](mailto:harsh25001@iiitd.ac.in)
- **Md. Shad Akhtar** — [shad.akhtar@iiitd.ac.in](mailto:shad.akhtar@iiitd.ac.in)

