// Knowledge base for the portfolio assistant.
// Each "## " section becomes one retrievable chunk. Keep facts here in sync with index.html.
// Files starting with "_" inside /api are not exposed as routes by Vercel.

module.exports = `
## Profile summary
Ashish Soni is an AI engineer and B.Tech student in Artificial Intelligence & Machine Learning at GGSIPU (Guru Gobind Singh Indraprastha University), New Delhi, India. He expects to graduate in 2028 and has a CGPA of 9.19 / 10.
He builds AI systems that "show their work": multi-agent pipelines, retrieval-augmented generation (RAG), computer vision, and the evaluations and tests that keep them honest. He reports numbers with their denominators and caveats.
He is currently an AI Development Intern at Think Decor, and previously completed a Back-End AI Engineering internship at FlyRank AI.
Portfolio: https://ashish-portfolio-sigma.vercel.app

## Availability and hiring
Ashish is open to AI/ML engineering internships and research collaborations, remote or on-site in India.
Roles he is interested in: AI/ML engineer intern, LLM / RAG / agentic AI engineer, applied AI engineer, ML research intern, computer vision intern.
He usually replies to email within a day. Best contact: ashishsoni243k@gmail.com.

## Contact and links
Email: ashishsoni243k@gmail.com
GitHub: https://github.com/ashishsoni-ai
LinkedIn: https://linkedin.com/in/ashish-soni-engineer
Résumé (PDF): https://ashish-portfolio-sigma.vercel.app/ashish_soni_resume.pdf
Location: New Delhi, India.

## Experience: Think Decor (current)
Role: AI Development Intern at Think Decor, an interior design company.
Dates: July 2026 to present. Remote, company based in the United Kingdom.
Work: building computer vision and AI features for Think Decor's products.
Skills: computer vision, artificial intelligence, Python.

## Experience: FlyRank AI (completed)
Role: Back-End AI Engineering Intern at FlyRank AI.
Dates: July 2026 to September 2026. Remote.
He was selected for the FlyRank AI internship programme and built backend AI services and APIs in Python: LLM-powered applications, API design and testing. He completed it and received a certificate of completion.
Skills: Python, agentic AI development, REST APIs, testing.

## Education
B.Tech in Artificial Intelligence & Machine Learning, GGSIPU, New Delhi, 2024 to 2028 (expected). CGPA 9.19 / 10.
Studies machine learning, deep learning and data systems.

## Project: ClauseGuard (featured)
Tags: agents, LLM evaluation, RAG, testing, featured project.
ClauseGuard (repo: PromiseCheck-clauseguard-) is a policy-conformance harness and deploy gate for money-touching customer-support agents, built for the Razorpay Open Track.
It runs 30 hand-written probes against frozen agents. Every failing row carries the policy clause and the sentence of the reply that contradicts it, both verified as literal substrings rather than paraphrased by a model.
Result: a naive 7B RAG agent made 11 over-promises in 30 probes; a 120B agent with structured retrieval made 2, an 82% reduction with no new failures. This is reported as a lower bound because the stronger agent shares a model family with the rule extractor.
Stack: Python, LLM-as-judge, RAG, structured retrieval.
Link: https://github.com/ashishsoni-ai/PromiseCheck-clauseguard-

## Project: HYDRA adversarial fraud lab (featured)
Tags: adversarial machine learning, fraud detection, anomaly detection, evaluation, featured project.
HYDRA is an adaptive adversarial fraud lab. It breeds a diverse population of synthetic payment-fraud behaviours that target what a detector cannot see, maps the detector's blind spots as a 120-cell atlas, hardens the detector, and tests whether the hardening transfers to an attack family it has never seen, all at a constant false-positive budget.
It has 680 passing tests and every reported figure traces to a committed results file. It runs only in a simulator over synthetic data; fidelity scores are calibrated to the simulator, not real card traffic.
Stack: Python, quality-diversity search, anomaly detection, pytest.
Live demo: https://hydra-fraud-lab.vercel.app  Repo: https://github.com/ashishsoni-ai/Hydra

## Project: AI Data Analyst (featured)
Tags: agents, LangGraph, RAG, retrieval, data analysis, OCR, full-stack, featured project.
AI Data Analyst: upload a CSV or Excel file and get automatic cleaning plus an interactive Power BI style dashboard with KPIs, filters and AI insights; upload PDFs, DOCX or images and chat with them using page-cited, grounded RAG.
A LangGraph router sends each question to SQL lookups, sandboxed statistics/ML, dataset overview or document Q&A. Numeric answers are consistency-checked and document answers cite their page. OCR handles scanned pages. Groq is the primary LLM with automatic Mistral failover via LiteLLM.
Stack: FastAPI, LangGraph, DuckDB, Qdrant, Celery, Redis, PyMuPDF, Tesseract, Next.js, TypeScript, Tailwind.
Link: https://github.com/ashishsoni-ai/ai-data-analyst

## Project: MedGuard
Tags: agents, LangGraph, computer vision, pose estimation, healthcare, research.
MedGuard is a five-agent LangGraph pipeline (Perception, Context Memory, LLM Reasoning, Decision, Action) to reduce false alarms in camera-based medical emergency detection. YOLOv8n-Pose extracts keypoints, a rolling 10-frame buffer tracks trajectories, and Gemini performs chain-of-thought reasoning. Asymmetric cost-sensitive thresholds minimise missed emergencies. It includes a 30-scenario evaluation harness. A research manuscript is in preparation.
Stack: LangGraph, YOLOv8-Pose, Gemini, FastAPI, Pydantic, OpenCV.
Link: https://github.com/ashishsoni-ai/MedGuard

## Project: AI Resume Screener
Tags: agents, LangGraph, LLM, NLP, backend.
A multi-agent LangGraph pipeline that parses PDF/DOCX resumes, extracts structured candidate information, analyses and scores skills, and writes a recruiter report. FastAPI backend, Streamlit frontend, Groq Llama 3.3.
Link: https://github.com/ashishsoni-ai/ai-resume-screener

## Project: AI Research Agent
Tags: agents, LangGraph, LLM, web research.
A stateful LangGraph research agent: Search (Tavily), Scrape (BeautifulSoup), Write (Mistral AI) and Critic nodes, with a conditional rewrite loop until the critic is satisfied, and a Streamlit frontend with live streaming.
Link: https://github.com/ashishsoni-ai/AI_Reasearch_Agent

## Project: PDF Chat Assistant
Tags: RAG, retrieval, LangChain, vector database, LLM.
A RAG app to chat with multiple PDFs: Unstructured parsing, AI-summarised chunks, HuggingFace all-MiniLM-L6-v2 embeddings, persistent ChromaDB vectors, Groq Llama 3.3 70B answers with chat history and source attribution, Gradio UI.
Link: https://github.com/ashishsoni-ai/pdfchat-groq-rag

## Project: Clinic Support Workflow
Tags: agents, LLM, customer support, Claude.
An AI customer-support workflow for a fictional aesthetics clinic, built for an AI engineering intern assessment at Closira: SOP-grounded FAQ answering, lead qualification, escalation detection (anger, medical questions, pricing, SOP gaps) and a structured JSON session summary. Python and the Claude API.
Link: https://github.com/ashishsoni-ai/closira-assignment

## Project: DeepLense GSoC 2027 tests
Tags: computer vision, deep learning, CNN, research, agents, Google Summer of Code.
Evaluation tests for ML4Sci DeepLense (Google Summer of Code 2027, agentic AI track). A multi-class CNN classifier for gravitational-lensing images (no substructure, sphere, vortex) reached 92.77% validation accuracy with per-class AUC of 0.98 to 0.99. He also built a Pydantic-validated agentic workflow around the DeepLenseSim simulation pipeline with natural-language parameter extraction and human-in-the-loop clarification, and documented a real upstream library incompatibility.
Link: https://github.com/ashishsoni-ai/deeplense-gsoc-2027

## Project: Speech Recognition (ASR) Pipeline
Tags: audio, speech recognition, deep learning, evaluation.
A reproducible pipeline that runs facebook/wav2vec2-base-960h on LibriSpeech samples from Hugging Face and reports word error rate (WER), character error rate (CER) and latency, all from a single command. Includes a research summary of wav2vec 2.0.
Link: https://github.com/ashishsoni-ai/speech-asr-pipeline

## Project: Deepfake Face Detector
Tags: computer vision, deep learning, CNN, image classification.
A CNN classifier for real versus AI-generated faces, comparing MobileNetV3Large, Xception and EfficientNetB4 with transfer learning and fine-tuning. TensorFlow and Keras.
Link: https://github.com/ashishsoni-ai/deepfake-face-detector

## Project: Image Captioning Model
Tags: computer vision, NLP, deep learning, CNN, LSTM.
An image-to-text model: Xception CNN encoder (ImageNet) and LSTM decoder trained on Flickr8k (8,000 images, 40,000 captions), with beam search at inference. TensorFlow and Keras.
Link: https://github.com/ashishsoni-ai/Image_Captioning_Model

## Project: Audio Genre Classification
Tags: audio, deep learning, music, classification.
Music genre classification from MFCC and mel-spectrogram features with neural networks. Librosa, TensorFlow.
Link: https://github.com/ashishsoni-ai/Audio_Genre_Classification

## Project: OpsLens data health dashboard
Tags: machine learning, anomaly detection, backend, full-stack, data quality.
OpsLens (repo ai-data-health-detector) is an ML-powered CSV audit tool: Isolation Forest anomaly detection, flags missing, stale and placeholder values, and exports cleaned, flagged and anomalous rows. FastAPI and React, Docker.
Link: https://github.com/ashishsoni-ai/ai-data-health-detector

## Project: meshery-ai, PromptPin, Social Media Studio
Tags: backend, full-stack, web apps, Go, LLM.
meshery-ai: an early Go adapter that connects the Meshery cloud-native manager to local LLMs (Ollama) over gRPC for natural-language infrastructure workflows.
PromptPin: a Pinterest-style board for AI image prompts with LLM prompt remixing; Next.js, FastAPI, Supabase, Cloudflare R2, Groq.
Social Media Studio: a Django app for scheduling posts across eleven social platforms with a calendar and dashboard, deployed on Render with PostgreSQL.
Links: https://github.com/ashishsoni-ai/meshery-ai  https://github.com/ashishsoni-ai/promptpin  https://github.com/ashishsoni-ai/Social-Media-Studio

## Open-source contributions
Ashish has 19 merged pull requests in other people's open-source projects: 15 in c2siorg/TensorMap, plus NNPDF/eko, Logara-AI (2) and CricScope.
TensorMap highlights: Keras 3 compatibility for SavedModel and TFLite export (#412, #413), validation of hyperparameter-tuning search spaces (#407), rejecting disconnected or malformed canvas graphs instead of crashing (#392, #401), deterministic ordering for the training-jobs endpoint with a de-flaked test (#405), SQLite-compatible migrations, and several missing unit tests.
NNPDF/eko #570: pass plain ints instead of enums to Numba kernels. Logara-AI: normalise Unix epoch timestamps; deep-copy on redaction. CricScope: ball-by-ball CSV export for win-probability predictions.
In review: TensorMap #419 static graph analysis with live shapes, Meshery #20399 AI adapter design spec and unit tests, DeepChem #5075 loss-argument fix, mesa-llm #328 memory fix, JdeRobot PerceptionMetrics tutorial fixes, and four InVesalius crash fixes.
All PRs: https://github.com/search?q=author%3Aashishsoni-ai+type%3Apr+-user%3Aashishsoni-ai&type=pullrequests

## GitHub activity
Ashish made about 650 public GitHub contributions in the last year, with a longest streak of 76 days (30 June to 13 September 2026) and around 150 active days. He also practises data structures and algorithms daily in his LeetCode_Solutions repository. The portfolio shows a live contribution calendar.

## Skills and toolkit
Agents and LLMs: LangGraph, LangChain, RAG and retrieval, LLM evaluation, tool calling, prompt engineering, Groq, Gemini, Mistral, Claude, ChromaDB, Qdrant, FAISS.
Machine learning: PyTorch, TensorFlow, Keras, scikit-learn, CNNs, LSTMs, Transformers, transfer learning, YOLOv8, OpenCV, Hugging Face, audio ML (MFCC, wav2vec2).
Backend: Python (primary language), FastAPI, Django, PostgreSQL, DuckDB, SQLite, Celery, Redis, Docker, pytest.
Also: C, C++, Go (basics), TypeScript, Next.js, React, Streamlit, Gradio, Git, Vercel, Render, Hugging Face Spaces.

## How Ashish works
He focuses on complete systems rather than just training models, and on measuring them honestly: frozen agents under test, literal-substring evidence, committed result files, and documented limitations. He learns codebases by contributing small, unglamorous fixes: compatibility issues, input validation, flaky tests and missing coverage.

## About this assistant
This chat assistant runs on the portfolio itself. It retrieves the most relevant sections of Ashish's portfolio and answers from them using a free-tier language model (Groq, running open models such as GPT-OSS, with Gemini as a backup). If no model is reachable, it answers directly from the retrieved portfolio text.
`;
