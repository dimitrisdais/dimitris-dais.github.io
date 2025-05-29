---
title: "Demystifying Summarization Models: How to Choose the Right Tool for the Job"
excerpt: "Text summarization is no one-size-fits-all. This blog unpacks how different language models—from compact classics to massive chatbots—handle summarization, and what techniques help scale to longer inputs."
date: 2025-05-28
layout: single
author_profile: true
read_time: true
toc: true
toc_sticky: true
categories:
  - NLP
  - LLM
tags:
  - summarization
  - language-models
  - generative-ai
  - model-comparison
  - practical-guide
---

In the current language model ecosystem, text summarization plays a central role in many applications—from streamlining documents and academic literature to extracting key insights from lengthy reports. As the demand for summarization increases, so does the number of models and tools available to perform the task.

From classic encoder-decoder architectures to large-scale chat-optimized models, the range of available options is extensive. Each model comes with distinct strengths, limitations, and underlying assumptions, which makes the process of selecting the appropriate tool a non-trivial task.

This blog provides a structured overview of summarization models across different scales and design philosophies. It begins with foundational models such as BART, continues with specialized models finetuned for longer contexts, and concludes with general-purpose chat-based models like ChatGPT. Along the way, the discussion highlights key trade-offs, summarizes model behavior, and outlines practical strategies for handling longer input texts.

![AI mastering the art of summarization.](https://raw.githubusercontent.com/dimitrisdais/dimitris-dais.github.io/master/assets/img/robot_learning_to_summarize.png)

📘 You can explore the full pipeline here: [Full Notebook: Summarization Task](https://github.com/dimitrisdais/generative-ai-lab/blob/main/notebooks/summarization_task.ipynb)

This post is part of the [**Generative AI Lab**](https://github.com/dimitrisdais/generative-ai-lab), a public repository that explores creative and practical uses of LLMs, Vision-Language Models (VLMs), and multimodal AI pipelines. If you are curious about how these systems can be chained together to automate creative workflows, feel free to explore the other blogs in the repository to discover more exciting applications.

---

## 📚 Table of Contents

- [📉 BART as a Summarization Baseline](#-bart-as-a-summarization-baseline)
- [🔍 Finetuned Models for Longer Contexts and Structured Abstraction](#-finetuned-models-for-longer-contexts-and-structured-abstraction)
- [🧠 Scaling Further: General-Purpose Open-Source LLMs (Qwen and Mistral)](#-scaling-further-general-purpose-open-source-llms-qwen-and-mistral)
- [💬 Proprietary Chat Models: ChatGPT and ChatGPT Mini](#-proprietary-chat-models-chatgpt-and-chatgpt-mini)
- [🧭 Conclusion and Takeaways](#-conclusion-and-takeaways)

---

## 📉 BART as a Summarization Baseline

BART (Bidirectional and Auto-Regressive Transformers) is a well-established baseline for abstractive summarization. Developed by Facebook AI, BART is a **sequence-to-sequence transformer architecture** that combines the bidirectional encoding of BERT with the left-to-right decoding of GPT. It was pretrained using a **denoising autoencoding objective** and later fine-tuned for summarization tasks, most notably on the CNN/DailyMail dataset. A popular implementation is available as [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) ([Lewis et al., 2020](https://arxiv.org/abs/1910.13461)). BART is widely used as a research and production baseline in the summarization space. For instance, it has been integrated into platforms like **Hugging Face Transformers** and served as a benchmark model in evaluations such as **SummEval** ([Fabbri et al., 2020](https://arxiv.org/abs/2007.12626)).

One known limitation of BART—particularly relevant in real-world use cases—is its **input size constraint**. The `facebook/bart-large-cnn` model can process up to **1024 tokens**, which typically corresponds to around **700–800 words**. This restriction means that any moderately long document must be **chunked** or truncated before summarization, which introduces challenges in maintaining coherence and context.

To evaluate BART’s performance, the model was applied to text extracted from a personal webpage containing a detailed professional profile: [dimitrisdais.github.io](https://dimitrisdais.github.io/dimitris-dais.github.io/). The content included biographical information, project experience, technical skills, and tool familiarity—structured in a typical curriculum vitae format.

The following excerpt represents a sample of the input text:

> - Dimitris Dais is a Senior Machine Learning Engineer with a PhD in Artificial Intelligence and Civil Engineering.  
> - He has built and deployed cloud-based AI pipelines for real-time visual understanding and automated incident verification using Vision-Language Models (VLMs). Dimitris has led the R&D of cutting-edge solutions for automatic industrial inspections and seismic damage assessment.  
> - He also led a project applying AI to detect and monitor cracks on buildings under earthquake loads.  
> - Programming & ML Frameworks: Python, PyTorch, TensorFlow, Keras, scikit-learn, ultralytics, Hugging Face, OpenAI.  
> - Vision-Language Models (VLMs), transformers, LLM APIs, prompt engineering, zero/few-shot learning.  
> - Retrieval-Augmented Generation (RAG) & Q&A: FAISS, SentenceTransformers, LlamaIndex, LangChain.  
> - 3D Reconstruction: OpenCV, COLMAP, Open3D, Metashape.

The resulting summary from BART displayed mixed quality. The initial sentences were generally coherent, reflecting an understanding of the subject’s professional background. However, the latter part of the output devolved into a flat listing of tools and frameworks, echoing the structure of the input rather than synthesizing or paraphrasing it. This behavior suggests that the **structure of the input**—especially when split into sections like biography, skills, and tools—significantly influences the summarization quality.

In natural language processing, text summarization approaches generally fall into two broad categories:

1. **Extractive Summarization**
   - Selects and **copies** important sentences or phrases directly from the input text.
   - No paraphrasing or rewording.
   - Think of it as a "smart highlighter."

2. **Abstractive Summarization**
   - **Generates** new sentences that **paraphrase or rephrase** the original content.
   - Uses its language generation ability to convey the key points in a more concise and fluent form.
   - This is more similar to how humans summarize.

BART is designed to perform **abstractive summarization**, but as seen in the example, its outputs often include extractive elements. This is due to several factors:

- **Decoding conservatism**, which reduces hallucination risk but limits paraphrasing.
- **Biases in training data**, which often favor lexical overlap with the source.
- The use of **ROUGE scores** for evaluation, which reward models for copying phrases rather than rewording.

In summary, BART is an effective and well-understood starting point for summarization. However, its limitations—in input length, context handling, and abstraction depth—make it less suitable for complex, long-form documents without additional preprocessing or augmentation strategies. The next section explores models that begin to address these gaps.

---

## 🔍 Finetuned Models for Longer Contexts and Structured Abstraction

While BART provides a strong starting point for summarization, its limited input size and occasionally shallow abstraction make it less suitable for complex or lengthy documents. In contrast, recent models like **PEGASUS-X** extend the transformer architecture to accommodate significantly longer contexts—allowing them to better preserve structure and meaning across larger spans of text.

Two such models evaluated here are:

- [`pszemraj/pegasus-x-large-book_synthsumm-bf16`](https://huggingface.co/pszemraj/pegasus-x-large-book_synthsumm-bf16)
- [`BEE-spoke-data/pegasus-x-base-synthsumm_open-16k`](https://huggingface.co/BEE-spoke-data/pegasus-x-base-synthsumm_open-16k)

These models are based on the **PEGASUS-X** architecture, a long-context extension of PEGASUS, which was originally designed by Google for abstractive summarization through pretraining on a gap-sentence-generation task. PEGASUS-X incorporates architectural improvements that allow it to efficiently scale to input lengths of 16,000 tokens or more—making it ideal for multi-section documents.

---

### Why Finetuning Matters

The performance of these models is largely attributed to the datasets they were finetuned on:

- **`pszemraj/pegasus-x-large-book_synthsumm-bf16`** was finetuned on a synthetic dataset containing long-form text across multiple domains paired with summaries generated by GPT-3.5-turbo. It also includes "random" long-context examples drawn from general pretraining corpora. This diverse, instruction-rich data enables the model to generalize well and synthesize complex profiles.

- **`BEE-spoke-data/pegasus-x-base-synthsumm_open-16k`** is a lighter-weight variant, also finetuned on synthetic summaries for long inputs. While not as large or instruction-rich as the previous model, it offers solid performance with reduced resource requirements.

Both models accept **instruction-style prompts**, such as:
> _"Summarize the following professional profile in 2–3 sentences:\n\n"_

This prompt format improves output quality by explicitly steering the model’s generation process—a technique known as **instruction tuning**. It reduces ambiguity and often leads to more targeted, fluent summaries.

---

### Results and Observations

**`BEE-spoke-data/pegasus-x-base-synthsumm_open-16k`** generated the following:

> Dimitris Dais, a Senior Machine Learning Engineer with a PhD in Artificial Intelligence and Civil Engineering, has worked on real-time detection models and decision-support systems for defense and civil protection applications in Athens, Greece, London, UK, Zurich, Switzerland, and Rotterdam, The Netherlands.  
> He has a strong background in multimodal & generative AI, including VLMs/LLMs, transformers, prompt engineering, zero/few-shot learning, and cloud-based AI pipelines for real-time visual understanding and automated incident verification.

The output is **compact yet specific**, capturing geographical context, technical domains, and a thematic summary of expertise. However, it slightly underplays career breadth—perhaps due to conservative decoding or underutilized instruction signal.

**`pszemraj/pegasus-x-large-book_synthsumm-bf16`** returned a more detailed multi-sentence summary:

> Dimitris is a senior machine learning engineer with a PhD and extensive experience in AI and civil engineering.  
> He has developed and deployed AI solutions for defense, civil protection, industrial inspections, earthquake engineering, and 3D reconstruction.  
> He is actively seeking new opportunities and has experience in multimodal and generative AI, VLMs, and LLMs.  
> His professional profile includes experience in remote and freelance remote machine learning engineers, leading AI projects, and skills in programming and ML frameworks.

This result exhibits a **richer abstraction**, adding inferred elements such as career objectives and role types, likely drawn from patterns seen in the finetuning data. The summary retains fluency and organization across multiple dimensions, showing how longer-context models can maintain semantic continuity across sections of diverse content.

Both models outperform BART on longer, structured text. Their ability to handle extended input and follow instructions leads to more complete and fluent summaries. Next, we look at how larger general-purpose models compare.

---

## 🧠 Scaling Further: General-Purpose Open-Source LLMs (Qwen and Mistral)

Beyond summarization-specific models, general-purpose large language models (LLMs) like **Qwen** and **Mistral** offer powerful summarization capabilities as a byproduct of their broader instruction-tuned design. These models are open-source, self-hostable, and often optimized for versatility across a wide range of tasks—including summarization, classification, reasoning, and dialogue.

Unlike BART or PEGASUS-X, these models are not fine-tuned specifically for summarization, but their large capacity and diverse pretraining allow them to handle longer and more nuanced inputs effectively. They typically require more compute but offer higher abstraction quality and better generalization—especially when used with clear, instruction-style prompts.

Models Evaluated:

- [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B): a 8B-parameter instruction-tuned model from Alibaba, supporting multi-turn tasks and strong summarization performance.
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3): a compact, high-performance 7B model optimized for following instructions and multi-task use.

---

### Qwen3-8B

Qwen generated a summary that was detailed and well-written, but it **repeated itself unnecessarily**, providing multiple versions of the same content:

> Dimitris Dais is a Senior Machine Learning Engineer with a PhD in Artificial Intelligence and Civil Engineering, specializing in end-to-end AI solutions for complex real-world challenges.  
> He has extensive experience in multimodal and generative AI, including Vision-Language Models, transformers, and zero/few-shot learning, alongside hands-on deployment of cloud-based AI pipelines for real-time visual understanding.  
> His career spans both academic research and industry applications, with a focus on AI innovation in areas such as structural health monitoring, industrial inspections, and defense-related systems.  
> _(Note: The model then proceeded to produce a shorter version, followed by another, repeating itself multiple times.)_

This repetition suggests that the model may **struggle with strict format constraints** (e.g., "3 sentences only") unless explicitly reinforced in the prompt or post-processed. Such issues can often be mitigated with stronger prompt conditioning or length penalties, but they highlight a limitation when precision formatting is required.

---

### Mistral-7B-Instruct

The response from Mistral was concise, focused, and arguably the most effective overall:

> Dimitris Dais is a Senior Machine Learning Engineer with a PhD in Artificial Intelligence and Civil Engineering.  
> He has extensive experience in delivering end-to-end AI solutions for complex, real-world challenges, including defense, civil protection, vehicle detection, and industrial inspections.  
> Skilled in defining problem scope, data strategy, and selecting optimal AI stacks, Dimitris has a proven track record in AI innovation, with a focus on multimodal and generative AI, including Vision-Language Models, transformers, prompt engineering, zero/few-shot learning, and Retrieval-Augmented Generation.  
> He has a strong background in computer vision and 3D reconstruction, as well as development tools, backend & deployment, and cloud/DevOps.

This output reflects an understanding of the entire input. It **accurately captures elements that appear near the end of the document**, such as cloud/DevOps and backend tooling—suggesting that Mistral was able to retain and integrate information across the full context window.

Its performance illustrates how even smaller models, when well-instructed and properly optimized, can produce high-quality, structured summaries with minimal prompt engineering.

---

### Summary

Mistral and Qwen show that open-source instruction-tuned models can deliver strong summarization results—even without task-specific fine-tuning. While Qwen offers more capacity, it may need stricter guidance to avoid verbose or redundant outputs. Mistral, despite being lighter, provided one of the most coherent and balanced summaries, handling both structure and content with precision.

Next, the comparison turns to **proprietary models like ChatGPT**, which offer even more power—but at the cost of openness, deployability, and control.

---

## 💬 Proprietary Chat Models: ChatGPT and ChatGPT Mini

The final group evaluated includes **proprietary, instruction-tuned chat models**: OpenAI’s GPT-4o (the new default ChatGPT model) and GPT-4.1 Mini (a lightweight variant sometimes referred to informally as “ChatGPT Mini”). These models are not open-source and cannot be self-hosted, but they deliver state-of-the-art performance in a wide range of tasks—including summarization.

These chat-based models benefit from extensive reinforcement learning with human feedback (RLHF), large-scale instruction tuning, and infrastructure-level optimization. As a result, they tend to produce the most fluent, coherent, and contextually aware summaries—even with minimal prompt engineering.

However, these benefits come at a cost: usage depends on paid APIs, model internals are not transparent, and local deployment is not possible. This makes them less suitable for privacy-sensitive or offline workflows, despite their capabilities.

---

### GPT-4o

The summary produced by GPT-4o is polished, fluent, and captures not only factual content but also tone and positioning:

> Dimitris Dais is a Senior Machine Learning Engineer with a PhD in Artificial Intelligence and Civil Engineering, combining academic excellence and over a decade of technical experience to deliver robust AI systems across domains such as defense, civil protection, and industrial inspection.  
> He specializes in multimodal and generative AI, particularly Vision-Language Models, and has designed and deployed real-time pipelines leveraging zero-shot learning, cloud infrastructures, and advanced 3D computer vision techniques.  
> His work, supported by a strong publication record and leadership in interdisciplinary teams, consistently bridges cutting-edge research with impactful, real-world solutions.

This is arguably the **most refined and human-like summary** across all models tested. It captures achievements, scope, and nuance in a way that feels editorial rather than mechanical. Notably, it integrates long-tail information like “publication record” and “interdisciplinary teams” from scattered parts of the input.

---

### GPT-4.1 Mini

The GPT-4.1 Mini model—while lighter—still produced a high-quality, structured summary:

> Dimitris Dais is a Senior Machine Learning Engineer with a PhD specializing in AI-driven solutions for complex real-world problems, demonstrating expertise in multimodal and generative AI, including Vision-Language Models and transformers.  
> He has a strong record of delivering end-to-end systems across diverse sectors such as defense, civil protection, and industrial inspection, combining academic research excellence with practical deployment of scalable AI pipelines.  
> His leadership in R&D, project lifecycle management, and innovation is underscored by numerous publications, project funding, and cross-functional collaboration in both academia and industry.

While slightly more compressed than GPT-4o, the summary retains depth and fluency. It balances abstract positioning (e.g., “leadership in R&D”) with concrete accomplishments, making it suitable for executive or evaluative use cases.

---

### Comparison with Open Models

Compared to open-source models like Mistral and Qwen, the GPT-4-based chat models offer:

- **Higher fluency and polish**, with better handling of tone and implied context.
- **Consistent summary synthesis** even across long, multi-part documents.
- **Minimal need for prompt tuning** or output postprocessing.

However, they also present clear trade-offs:

- **Cost**: These models are API-based and billed per token, unlike local open-source models.
- **Deployment**: Cannot be self-hosted; unsuitable for restricted environments.
- **Transparency**: Model architecture and training details are not publicly available.

GPT-4o and GPT-4.1 Mini demonstrate what is possible when model capacity, training scale, and instruction tuning converge. Their outputs were the most fluent and contextually integrated—but at the expense of openness and control. For tasks where maximum summary quality is essential and infrastructure allows for API use, these models clearly lead the field.

---

## 🧭 Conclusion and Takeaways

Summarization is not a one-size-fits-all task. It demands thoughtful model selection based on input size, abstraction needs, resource availability, and deployment constraints.

- **BART** remains a reliable starting point but struggles with longer inputs.
- **PEGASUS-X models** offer major improvements for structured, long-form content—especially when instruction-tuned.
- **Open-source LLMs** like Qwen3-8B and Mistral-7B-Instruct strike a strong balance between flexibility and control, with Mistral standing out for coherence and coverage.
- **Proprietary models** such as GPT-4o and GPT-4.1 Mini currently lead in fluency and summary quality but at the cost of openness and deployability.

| Model                            | Type               | Max Input Length | Summary Style          | Open-Source | Deployment     |
|----------------------------------|--------------------|------------------|-------------------------|-------------|----------------|
| **BART**                         | Baseline           | ~1,024 tokens    | Shallow abstractive     | ✅ Yes      | Local / Cloud  |
| **PEGASUS-X (base)**             | Finetuned (16k)    | 16,000+ tokens   | Structured, concise     | ✅ Yes      | Local / Cloud  |
| **PEGASUS-X (large)**           | Finetuned (rich)   | 16,000+ tokens   | Detailed, inferred      | ✅ Yes      | Local / Cloud  |
| **Mistral-7B-Instruct**          | General-purpose LLM| ~32,000 tokens   | Precise, balanced       | ✅ Yes      | Self-hostable  |
| **Qwen3-8B**                     | General-purpose LLM| ~32,000 tokens   | Verbose, expressive     | ✅ Yes      | Self-hostable  |
| **GPT-4.1 Mini**                 | Proprietary Chatbot| High (estimated) | Polished, compressed    | ❌ No       | API-only       |
| **GPT-4o (default ChatGPT)**     | Proprietary Chatbot| High (estimated) | Editorial, nuanced       | ❌ No       | API-only       |

Ultimately, choosing the right summarization model means understanding the trade-offs—between control and convenience, structure and synthesis, cost and quality—and aligning those with the specific demands of your task.

Thanks for reading — I hope you found it useful and insightful.  
Feel free to share feedback, connect, or explore more projects in the [Generative AI Lab](https://github.com/dimitrisdais/generative-ai-lab).
