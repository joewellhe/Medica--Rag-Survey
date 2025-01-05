# A Survey on Medical and Healthcare RAG: Recent Advances and New Frontiers

This repository provides a comprehensive survey on the application of Retrieval Augmented Generation (RAG) in the medical and healthcare domains. We present the basic framework of medical RAG, detailing its commonly used components, datasets, and evaluation methods. Additionally, we've compiled a collection of state-of-the-art (SOTA) approaches and highlighted some literature that explores new frontiers in this field. We are committed to regularly updating this repository and welcome any feedback or suggestions.

## Introduction

With the emergence of large language models (LLMs) in recent years, numerous natural language processing (NLP) tasks have seen remarkable advancements. Their impressive capabilities in generating and understanding human-like text have resulted in outstanding performance in tasks such as summarization, question answering, information retrieval, and more. The exceptional performance of LLMs in core NLP tasks is prompting their exploration in the medical domain, ranging from aiding clinicians in making more accurate decisions to enhancing patient care quality and clinical outcomes. 

However, LLMs often generate plausible-sounding but factually incorrect responses, a phenomenon commonly known as hallucination. Additionally, once the training process is complete, the parameters of LLMs are fixed, resulting in a lack of up-to-date knowledge. Retrieval Augmented Generation (RAG) has the potential to alleviate these critical challenges because it can provide the rationale behind its generation and readily access the latest knowledge.

This survey focuses on useful techniques and the latest advances in medical and healthcare RAG. We first illustrate its basic framework and important components, and then we detail some useful improvements to these components separately. Next, we introduce datasets commonly used to evaluate medical and healthcare RAG, along with widely used knowledge sources. Finally, we present some evaluation metrics commonly used in experiments and explore new frontiers in this field.

(Please note that these new frontiers are constantly evolving. We strive to stay updated with the latest work and welcome any suggestions.)

## Basic framework

Here we present a basic framework of the vanilla medical RAG. As shown in the following figure, there are four key components in medical RAG: the retriever, knowledge source, ranking method, and large language model (LLM). A question is first processed by the retriever, which indexes some relevant documents from a variety of knowledge sources composed of webpages, academic papers, textbooks and so forth. After retrieval, we obtain references, also referred to as context or background knowledge in some literature. RAG uses ranking methods to sort these references based on their relevance to the original question. Finally, the top-k relented references, along with the original question, are sent to the LLM as input to generate the final result.

<img src=".\img\healthcare_Rag.png" alt="healthcare_Rag" />

## Retriever

The retriever is a key component to decide the relevance of references to the question. A good retriever can identify the most relevant and useful documents to answer the question, while a poor one may fail to be helpful and introduce noisy information.  Here we divide these retrievers into following 3 different types. 

### Lexical Retriever

**BM25** [[pdf]](https://dl.acm.org/doi/abs/10.1561/1500000019) is a ranking function used in information retrieval to estimate the relevance of documents to a given search query. It is commonly treated as a baseline for comparison with other retrievers. However, in many tasks, experimental results demonstrate that it still offers competitive performance.

### Search Engine Retriever

Using a search engine provides access to a wide range of external knowledge sources, making the search engine retriever a promising component in RAG (Retrieval-Augmented Generation). Below, we list some tools that are commonly used as retrievers in medical RAG, along with relevant literature that utilizes these tools.

#### NCBI Tool

> The [National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/) (NCBI) provides many useful products, including [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/), [PubChem](https://pubchem.ncbi.nlm.nih.gov/), [Gene](https://www.ncbi.nlm.nih.gov/gene), and [Genome](https://www.ncbi.nlm.nih.gov/data-hub/genome/). In addition to the web interfaces to these products, NCBI also provides an API allowing programmatic access to the underlying databases and search technology.

[**Entrez API**](https://www.ncbi.nlm.nih.gov/home/develop/api/), also known as Entrez Programming Utilities (E-utilities), is a set of web-based tools provided by the National Center for Biotechnology Information (NCBI). These tools allow researchers and developers to access and retrieve data from NCBI's comprehensive suite of biological databases programmatically.

**PubMed API** provides access to the PubMed database when you specify the database as "PubMed" in your search query. Note that the PubMed API is part of the Entrez API system. You can also specify other databases, such as PubMed Central or Gene, in your search queries.

#### Wikipedia Tool

[**Wikipedia API**](https://www.mediawiki.org/wiki/API:Main_page) is a set of application programming interfaces (APIs) that allows developers to access and interact with Wikipedia's vast content programmatically.

#### Question2Query

Sometimes, an LLM (Large Language Model) is used to transform a user's question or dialogue history into a search engine query, which is then executed in the search engine database. We refer to this method as 'Question2Query.' This approach is often used in combination with a search engine retriever.

#### Literature

- An open-source retrieval-augmented large language model system for answering medical questions using scientific literature. [[pdf]](https://psb.stanford.edu/psb-online/proceedings/psb24/lozano.pdf) <br>using Entrez API as retriever, Question2Query <br>https://github.com/som-shahlab/Clinfo.AI/tree/main
- Tool calling: Enhancing medication consultation via retrieval-augmented large language models.[[pdf]](https://arxiv.org/html/2404.17897v1) <br>Distilling the key information and forming the searching query (Question2Query ), using search engine as retriever

### Semantic Retriever

Due to recent advancements in deep learning, semantic retrievers, also known as dense retrievers, have achieved impressive performance and are widely used in Biomedical RAG. These retrievers encode and match queries and documents as dense vectors (document embeddings). This approach often utilizes Pre-trained Language Models (PLMs) to encode documents, treating the nearest documents in vector space as the most relevant at a semantic level. 

#### [Vector Stores](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/)

> One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.

Vector stores are an important component in semantic retrievers, offering efficient search methods like K-nearest neighbors (KNN) for RAG developers. They enable rapid retrieval of semantically similar documents, enhancing the performance of Biomedical RAG systems. The process of embedding, storing, and searching documents is illustrated in the following picture. Here, we list two commonly used vector stores in Biomedical RAG.

<img src=".\img\vector_stores.jpg"/>

> [**Chroma**](https://docs.trychroma.com/docs/overview/getting-started) is an AI-native open-source vector database. It comes with everything you need to get started built in, and runs on your machine.

> [**Faiss**](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. 

#### Embedding-based Retriever

Also called dense retriever. We can construct a semantic retriever by combining an embedding method with a vector store. However, it is crucial to select an appropriate embedding model, as the training corpora of these models vary, leading to differing abilities to encode various types of documents.

##### General Embedding Models

General embedding models are trained on general corpora and are widely used in various information retrieval systems. There are a substantial number of open-source general embedding models, and they are often treated as baselines in Biomedical RAG experiments. The following table shows some representative models.

| Model        | Feature                | Data | Link                                                         |
| ------------ | ---------------------- | ---- | ------------------------------------------------------------ |
| **LDA**      | Machine Learning based | 2003 | [pdf](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf); [Github](https://github.com/lda-project/lda) |
| **Doc2Vec**  | Deep Learning based    | 2014 | [pdf](https://proceedings.mlr.press/v32/le14.html); [Github](https://github.com/inejc/paragraph-vectors) |
| **FastText** | Deep Learning based    | 2017 | [pdf](https://aclanthology.org/Q17-1010.pdf); [Github](https://github.com/facebookresearch/fastText/) |
| **Sent2Vec** | Deep Learning based    | 2018 | [pdf](https://aclanthology.org/N18-1049/); [Github](https://github.com/epfml/sent2vec) |
| **RoBERTa**  | BERT based             | 2019 | [pdf](https://arxiv.org/abs/1907.11692); [Hugging Face](https://huggingface.co/docs/transformers/model_doc/roberta) |
| **ColBERT**  | BERT based             | 2020 | [pdf](https://dl.acm.org/doi/10.1145/3397271.3401075); [Github](https://github.com/stanford-futuredata/ColBERT); [Hugging Face](https://huggingface.co/colbert-ir/colbertv2.0) |
| **SimCSE**   | Contrastive Learning   | 2021 | [pdf](https://aclanthology.org/2021.emnlp-main.552/); [Github](https://github.com/princeton-nlp/SimCSE) |

##### Commercial Embedding Models

Thanks to recent advances in Large Language Models, many AI companies now provide commercial embedding APIs, which are popular among biomedical researchers and developers. Although these services may be costly, especially with large datasets, their attractive performance and convenience (call API only, not need for train) has led to widespread use.  The following table lists some popular commercial embedding models. Note that each company offers models of various sizes, so the maximum input and embedding dimensions may vary.

| Model              | Max Input Token | Dimension | Company | Link                                                         |
| ------------------ | --------------- | --------- | ------- | ------------------------------------------------------------ |
| **text-embedding** | 8191            | 1536-3072 | OPEN AI | [document](https://platform.openai.com/docs/guides/embeddings) |
| **voyage-2**       | 4000~1600       | 1024-1536 | Claude  | [document](https://docs.anthropic.com/en/docs/build-with-claude/embeddings) |
| **Vertex AI**      | 3072            | 768       | Google  | [document](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) |
| **bge-large**      | 512             | 1024      | Baidu   | [document](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/mllz05nzk) |
| **tao-8k**         | 8192            | 1024      | Baidu   | [document](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/7lq0buxys) |

##### Embedding with open source LLMs

Due to their larger number of parameters, large language models have a superior ability to understand text. Some researchers choose to use the embedding layers of open-source LLMs for document embedding. There are many LLMs available for embedding purposes. More details about LLMs in RAG can be found in [Generation Model](#generation-model) section. Here, we provide an [example](https://github.com/ToneLi/BIoMedRAG/blob/main/0_make_relation_chuck_and_scorer_data/1_get_store_chuck_embeddings_5.py) that uses MedLLaMA as an embedding model.

##### Biomedical Embedding Models

In many biomedical NLP tasks, language models trained on biomedical-related corpora outperform general-domain language models due to their superior ability to understand biomedical language. Therefore, using a biomedical language model as an embedding model is a common approach to building a semantic retriever. Below, we list some popular biomedical embedding models.

| Model          | Base | Date | Link                                                         |
| -------------- | ---- | ---- | ------------------------------------------------------------ |
| **BioBERT**    | BERT | 2019 | [pdf](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506); [Github](https://github.com/naver/biobert-pretrained) |
| **PubMedBERT** | BERT | 2021 | [pdf](https://arxiv.org/abs/2007.15779); [Hugging Face](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| **UmlsBERT**   | BERT | 2021 | [pdf](https://aclanthology.org/2021.naacl-main.139.pdf); [Github](https://github.com/gmichalo/UmlsBERT) |
| **BioBART**    | BART | 2022 | [pdf](https://arxiv.org/abs/2204.03905); [Github](https://github.com/GanjinZero/BioBART) |

More information about biomedical embedding models can be found in Table I of this [survey](https://arxiv.org/abs/2310.05694).

### Recent Advanced Retriever

Although the aforementioned off-the-shelf retrievers are readily available, the task of searching for relevant and accurate documents in the medical domain remains challenging. Consequently, customized retrievers tailored specifically to these tasks have been developed. These advanced retrievers also achieve comparable performance. Here, we introduce some of them.

- **Contriever**:  An unsupervised general dense retriever based on contrastive learning. Experiments show that its zero-shot capability (trained on general corpora and applied to new domains) is promising. It is often used as a strong baseline in biomedical RAG.<br>[[pdf]](https://arxiv.org/abs/2112.09118); [[Github]](https://github.com/facebookresearch/contriever)

- **SPECTER**:  A model designed to generate document-level embeddings of scientific documents. It utilizes the citation graph as a training signal to capture inter-document relatedness, showing an advantage in generating document embeddings, particularly for scientific papers. In the biomedical domain, scientific papers are a vital part of external knowledge sources, so using SPECTER as a retriever in biomedical RAG demonstrates comparable performance.<br>[[pdf]](https://arxiv.org/pdf/2004.07180); [[Github]](https://github.com/allenai/specter)

- **MedCPT**: A specialized retriever trained on user click logs from PubMed. It includes a query encoder (QEnc), a document encoder (DEnc), and a ranking model (CrossEnc). The initial embedding model used in MedCPT is PubMedBERT. Contrastive loss is employed to train both the MedCPT retriever (QEnc, DEnc) and the MedCPT ret-ranker (CrossEnc).<br>[[pdf]](https://arxiv.org/abs/2307.00589); [[Github]](https://github.com/ncbi/MedCPT)
- **MEDRAG toolkit**:  A systematic implementation of RAG for medical QA. It provides a convenient way to search multi-source medical documents. Its retriever consists of four different components, including BM25, Contriever, SPECTER, and MedCPT.<br>[[pdf]](https://aclanthology.org/2024.findings-acl.372.pdf); [[Github]](https://github.com/Teddy-XiongGZ/MedRAG)

#### Literature

- BiomedRAG: A Retrieval augmented Large Language Model for Biomedicine [[pdf]](https://arxiv.org/abs/2405.00465) <br>Embedding with open source LLMs, using MedLLaMA 
- Zero-shot ecg diagnosis with large language models and retrieval-augmented generation [[pdf]](https://proceedings.mlr.press/v225/yu23b/yu23b.pdf)<br>Commercial Embedding Models, using  OPEN AI API
- Chatent: Augmented large language model for expert knowledge retrieval in otolaryngology-head and neck surgery [[pdf]](https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/full/10.1002/ohn.864)<br>Commercial Embedding Models, using  OPEN AI API
- Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented [[pdf]](https://arxiv.org/abs/2401.15269)<br>Recent Advanced Retriever, using MedCPT
- ReMAG-KR: Retrieval and Medically Assisted Generation with KnowledgeReduction for Medical Question Answering [[pdf]](https://aclanthology.org/2024.acl-srw.13.pdf)<br>Recent Advanced Retriever, using MedCPT

## Ranking Method

Actually, most of the aforementioned retrievers , e.g., BM25,  can provide a score for each retrieved document, allowing us to rank these documents based on their retrieval scores. However, for tasks in the biomedical domain, ranking may involve additional considerations. For instance, in the [2021 TREC Healthcare Misinformation Track](https://trec-health-misinfo.github.io/2021.html), a retrieved document is assessed based on three criteria: *Usefulness*, *Supportiveness*, and *Credibility*.

> *Usefulness*: The extent to which the document contains information that a search user would find useful in answering the topic’s question.<br>*Supportiveness*: Does the document support or dissuade the use of the treatment in the topic’s question<br>*Credibility*: whether the document is considered credible by the assessor.

 A good ranking method can accurately identify the rationale behind a user's query while filtering out noise and toxic information. Here, we introduce some practices.

#### Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) [[pdf]](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) is an algorithm that evaluates search scores from multiple, previously ranked result sets to produce a unified result set. RRF can be utilized when using multiple retrievers, and you want to combine their results into a single ranking. This approach gives higher weight to documents that are ranked highly in at least one list, even if they aren't ranked as highly in others. 

#### Trained Scorer & Distilled Scorer from LLM

Some choose to train a neural-network-based scorer model to rank retrieval results according to specific criteria. This approach requires an additional dataset to train a model that can evaluate documents based on different criteria. We introduce some datasets used to train a ranking scorer in the [Dataset](https://chat.chatbotapp.ai/chats/-OFoQ0KIui2T-mZwck6U?model=gpt-4#dataset) section.

Additionally, since large models like ChatGPT have strong evaluation capabilities, some developers opt to use large language models (LLMs) as scorers to filter relevant documents. However, because LLMs can be costly to run, distilling a model from an LLM can also be a good option.

#### Literature

- Benchmarking retrieval augmented generation for medicine [[pdf]](https://aclanthology.org/2024.findings-acl.372.pdf)<br>the aforementioned MEDRAG toolkit, using RRF to incorporate its four retrievers
- Online Health Search Via Multidimensional Information Quality Assessment Based on Deep Language Models: Algorithm Development and Validation [[pdf]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099810/pdf/ai_v3i1e42630.pdf) <br>trained scorer, utilizing BERT as the backbone model, using three additional datasets to evaluate documents based on *Usefulness*, *Supportiveness*, and *Credibility*, RRF is then used to obtain the final score.
- BiomedRAG：A retrieval augmented large language model for biomedicine [[pdf]](https://arxiv.org/abs/2405.00465)<br>distilled scorer from LLM, its Tailored Chunk Scorer is trained to align with the LLM.

## Generation Model

Generation, based on the references found by the retriever and the prompt, is a key step in providing the final response to the user. Large language models (LLMs) are often used as the generation component. There are a substantial number of LLMs available for biomedical RAG. We categorize these LLMs into three different types and list some widely used LLMs for each.

### General domain open source LLMs

| Model      | Para Size | Link                                                         |
| ---------- | --------- | ------------------------------------------------------------ |
| T5         | 0.06-11B  | [[Github]](https://github.com/google-research/text-to-text-transfer-transformer); [[Checkpoints]](https://huggingface.co/google/flan-t5-xxl) |
| ChatGLM3   | 6B        | [[Github]](https://github.com/THUDM/ChatGLM3/blob/main/README_en.md); [[Checkpoints]](https://huggingface.co/THUDM/chatglm3-6b) |
| OpenLLaMA  | 3,7,13B   | [[Github]](https://github.com/openlm-research/open_llama); [[Checkpoints]](https://huggingface.co/openlm-research/open_llama_7b) |
| LLaMA 2    | 7-70B     | [[Download Link]](https://www.llama.com/llama-downloads)     |
| LLaMA 3    | 8-70B     | [[Introduction]](https://ai.meta.com/blog/meta-llama-3/); [[Checkpoints]](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| MPT-7B     | 7B        | [[Introduction]](https://www.databricks.com/blog/mpt-7b); [[Checkpoints]](https://huggingface.co/mosaicml/mpt-7b) |
| Phi-3 Mini | 3.8B      | [[pdf]](https://arxiv.org/abs/2404.14219); [[Checkpoints]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Mistral 7B | 7B        | [[Introduction]](https://mistral.ai/news/announcing-mistral-7b/); [[Checkpoints]](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |

### General domain commercial LLMs

| Model | Company | Link |
| ----- | ------- | ---- |
|       |         |      |
|       |         |      |
|       |         |      |



### Biomedical domain open source LLMs



## Knowledge Source

## Dataset

## Evaluation Method

## Frontiers

