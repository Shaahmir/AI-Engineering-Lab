# AI-Engineering-Lab

A growing collection of hands-on AI engineering projects covering LLMs, AI agents, RAG, multimodal applications, model inference, and fine-tuning. Each project explores a specific concept through practical implementations designed for learning, experimentation, and developing real-world AI engineering skills.

## Folder Structure

The projects are organized roughly from introductory AI applications to more advanced techniques. Students can follow the sections in order or explore individual projects based on their interests.

### 1. AI Interfaces & Experiments

Start by exploring how AI systems can be presented through interactive interfaces and small experiments.

| Project | What it covers |
|---|---|
| [AI Interfaces in Gradio](./AI%20Interfaces%20in%20Gradio/) | Building interactive AI interfaces with Gradio. |
| [AI Logits Visualiser](./AI%20Logits%20Visualiser/) | Visualizing model logits and token-level behavior. |
| [Eavesdropping on AI](./Eavesdropping%20on%20AI/) | An experiment focused on observing and exploring AI behavior. |
| [Jev](./Jev/) | A small AI/LLM experiment. |

### 2. AI Applications & Agents

Explore practical AI applications, from summarizing websites to building assistants and agent-based workflows.

| Project | What it covers |
|---|---|
| [AI Website Summarizer](./AI%20Website%20Summarizer/) | Summarizing website content with an AI workflow. |
| [AI Website Summarizer (Local Models)](./AI%20Website%20Summarizer%20(Local%20Models)/) | Website summarization using locally run models. |
| [AI Brochure Generator](./AI%20Brochure%20Generator/) | Generating brochures with AI-assisted workflows. |
| [AI Flight Assistant](./AI%20Flight%20Assistant/) | AI-assisted flight search and travel workflows. |
| [AI Flight Assistant (Multimodal)](./AI%20Flight%20Assistant%20(Multimodal)/) | Exploring a multimodal flight-assistant workflow. |
| [AI Meeting Minutes Generator](./AI%20Meeting%20Minutes%20Generator/) | Turning meeting audio into structured meeting notes and minutes. |
| [AI Synthetic Data Generator](./AI%20Synthetic%20Data%20Generator/) | Experiments with generating synthetic data using AI models. |
| [AI Best Deal Finder Agent](./AI%20Best%20Deal%20Finder%20Agent/) | An agent-oriented workflow for finding and comparing product deals. |

### 3. LLM APIs, Tokenization & Inference

Learn how to interact with language models, process text into tokens, and run inference using hosted or local infrastructure.

| Project | What it covers |
|---|---|
| [LLAMA Tokenizer](./LLAMA%20Tokenizer/) | Exploring LLaMA-style tokenization and token data. |
| [OpenAI Batch API](./OpenAI%20Batch%20API/) | Exploring batch processing workflows with the OpenAI API. |
| [Inference on Modal](./Inference%20on%20Modal/) | Running model inference using Modal. |
| [OpenAI FineTune API](./OpenAI%20FineTune%20API/) | Exploring fine-tuning workflows through the OpenAI API. **(Deprecated)** |

### 4. Hugging Face

Explore Hugging Face tools and models, including Transformers pipelines, tokenizers, model architectures, quantization, and generative AI.

| Project | What it covers |
|---|---|
| [Hugging Face Transformers Pipelines](./Hugging%20Face/Hugging%20Face%20Transformers%20Pipelines.ipynb) | Using Transformers pipelines for common pretrained-model tasks. |
| [Visualising Tokenizers](./Hugging%20Face/Visualising%20Tokenizers.ipynb) | Exploring how tokenizers convert text into tokens. |
| [Visualising Architectures and Quantization](./Hugging%20Face/Visualising%20Architectures%20and%20Quantization.ipynb) | Visualizing model architectures and exploring quantization concepts. |
| [Hugging Face Image Generation](./Hugging%20Face/Hugging%20Face%20Image%20Generation.ipynb) | Exploring image generation using Hugging Face models and tools. |
| [Hugging Face Audio Generation](./Hugging%20Face/Hugging%20Face%20Audio%20Generation.ipynb) | Exploring audio generation using Hugging Face models and tools. |

### 5. RAG & Retrieval

Learn how to build Retrieval-Augmented Generation (RAG) systems that retrieve relevant information to support model responses. These projects explore different approaches to embeddings, vector databases, retrieval, reranking, and evaluation.

| Project | What it covers |
|---|---|
| [AI Simple RAG Implementation](./AI%20Simple%20RAG%20Implementation/) | A starting point for understanding a basic RAG workflow. |
| [AI Vector Databases](./AI%20Vector%20Databases/) | Exploring vector databases and similarity-based retrieval. |
| [AI RAG Implementation (Embeddings)](./AI%20Rag%20Implementation%20(Embeddings)/) | Exploring embeddings for retrieving relevant information. |
| [AI RAG with LangChain](./AI%20Rag%20with%20LangChain/) | Implementing RAG workflows with LangChain. |
| [AI RAG Implementation (ReRanking)](./AI%20Rag%20Implementation%20(ReRanking)/) | Exploring reranking to improve the ordering of retrieved results. |
| [AI RAG Evaluation](./AI%20Rag%20Evaluation/) | Evaluating RAG systems and the quality of their results. |

### 6. Fine-Tuning

Finally, explore how pretrained models can be adapted to specific tasks using fine-tuning. These notebooks cover fine-tuning workflows for Gemma and LLaMA models.

| Project | What it covers |
|---|---|
| [Finetuning Gemma - Unsloth](./Fine-Tuning/Finetuning%20Gemma%20-%20Unsloth.ipynb) | Fine-tuning Gemma models using Unsloth. |
| [Finetuning LLAMA](./Fine-Tuning/Finetuning%20LLAMA.ipynb) | Exploring the fine-tuning workflow for LLaMA models. |

## Running a Project

Projects may use different dependencies and entry points, so check the selected project's folder first.

For projects that include `pyproject.toml` and `uv.lock`, a typical setup is:

```bash
git clone https://github.com/Shaahmir/AI-Engineering-Lab.git
cd AI-Engineering-Lab
uv sync
```

Then open the project's notebook or run its Python entry point, as appropriate.
Some projects require API keys or environment variables. Configure these according to the instructions provided in the individual project.

## Learning Path

This repository brings together practical projects for learning AI engineering, from building AI-powered applications to exploring LLM internals, retrieval systems, and model fine-tuning. Each project provides an opportunity to experiment with tools, understand underlying concepts, and apply them in code.

Whether you're starting with AI interfaces or exploring more advanced techniques, the projects are organized to help you build your knowledge step by step.

## License

This repository is licensed under the [MIT License](./LICENSE).
