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

#### [Vector Store](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/)

> One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.

The process of embedding, storing, and searching documents is illustrated in the following picture. Here, we list two commonly used vector stores in Biomedical RAG.

<img src=".\img\vector_stores.jpg"/>

> [**Chroma**](https://docs.trychroma.com/docs/overview/getting-started) is an AI-native open-source vector database. It comes with everything you need to get started built in, and runs on your machine.

> [**Faiss**](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. 

#### Embedding-based Retriever

general embeddings model

commercial embeddings model

biomedical embeddings model

openAI的embeding

PubMedBERT

BioBERT

MedLLaMA 13b做embedding embedding层的平均

#### Scientific and Biomedical Retriever

Contriever

SPECTER

MedCPT

Literature

## Ranking Method

Reciprocal Rank Fusion

## Generation Model

## Knowledge Source

## Dataset

## Evaluation Method

## Frontiers

