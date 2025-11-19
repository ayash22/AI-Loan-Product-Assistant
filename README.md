# Loan Product Assistant – Bank of Maharashtra

## Project Overview

The **Loan Product Assistant** : An intelligent AI tool for Bank of Maharashtra loan products, leveraging a lightweight Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers to user queries quickly and efficiently.

## Key Features

- **Data Scraping & Processing**: Collects and cleans loan-specific information from the official Bank of Maharashtra website.
- **Knowledge Base Creation**: Structures data into high-quality chunks for semantic search and efficient retrieval.
- **Lightweight RAG Pipeline**: Uses FAISS vector store and an LLM to generate precise, context-aware responses.
- **Flexible Interfaces**: Supports both console-based querying and a full UI version for user convenience.
- **Practical AI Deployment**: Demonstrates real-world AI applications in banking, including RAG pipeline implementation and data handling.

The system leverages a structured knowledge base created from publicly available information about Bank of Maharashtra loan offerings. Users can ask questions such as:

- “What are the interest rates for a Bank of Maharashtra home loan?”
- “What is the maximum tenure for a personal loan if my salary account is with the bank?”
- “Tell me about the Maha Super Flexi Housing Loan Scheme.”
- “Are there any processing fee concessions for women or defence personnel on home loans?”

---

## Core Tasks Implemented

### Data Scraping & Collection
- Collected loan-specific information from Bank of Maharashtra’s official website.
- Focused on **personal loans, home loans, and special schemes**.
- Used structured scraping to ensure reliable and accurate data collection.

### Data Consolidation & Processing
- Cleaned and structured data to remove irrelevant content such as ads, navigation elements, or HTML tags.
- Consolidated loan information into a **single high-quality knowledge base** (`kb.txt` or `chunks.json`).
- Split text into chunks (200–300 words) to optimize semantic search for the RAG pipeline.

### Lightweight RAG Pipeline
- Implemented a simple RAG system using **FAISS** as an in-memory vector store.
- **Pipeline workflow:**
  1. Accept user question.
  2. Convert question to vector embedding.
  3. Retrieve most relevant text chunks from the knowledge base.
  4. Pass the retrieved context to an LLM for answer generation.
  5. Return the LLM-generated answer to the user.

---
## Configuration

To run the project, you need an API key for the LLM:

1. Go to [Groq Console](https://console.groq.com/home) and generate your API key.  
2. Open `src/rag/config.py` and enter your API key in the `LLM_API_KEY` variable:

```python
LLM_API_KEY = "your_api_key_here"
```

## Project Setup & Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run Locally (Console Version)

- Navigate to the src folder:
```commandline
cd src
```
- Run the scraper to collect data:

```bash
python scrapper.py
```
- Clean and chunk the scraped data:
```bash
python cleaner_and_chunker.py
```

- Navigate to the rag folder:
```commandline
cd rag
```

- Generate embeddings and create vector index:
```bash
python embed_index.py
- ```

- Start the RAG query interface:
```bash
python rag_query.py
```
- Enter questions in the console to get AI-generated answers.

3. Run Full UI Version
```bash
python main.py
```
## Architectural Decisions

### Libraries
- **Web Scraping:** `requests`, `BeautifulSoup` – structured and reliable scraping.  
- **Data Processing:** `pandas`, `nltk` – cleaning, structuring, and chunking text.  
- **RAG Pipeline:** `FAISS` – lightweight in-memory vector search.  
- **LLM:** Groq/OpenAI API integration for natural language generation.  

### Data Strategy
- Text data is chunked into **200–300 word segments** to optimize retrieval relevance and maintain context.  
- **Overlaps** are included between chunks to prevent information loss.  

### Model Selection
- **Embedding Model:** `all-MiniLM-L6-v2` – lightweight and effective for semantic search.  
- **LLM:** `openai/gpt-oss-20b` – generates accurate answers from retrieved content.  

### AI Tools Used
- **FAISS** for vector search.  
- **LLM API** for response generation.  
- **Scraping and cleaning scripts** for knowledge base preparation.  

### Description of Folders and Files

- **`.gitignore`** – Git ignore rules.  
- **`main.py`** – Entry point for the full UI version of the project.  
- **`README.md`** – Project documentation.  
- **`requirements.txt`** – Python dependencies.  

- **`.idea/`** – IDE configuration files (PyCharm).  

- **`data/`** – Contains all scraped and processed data.  
  - **`knowledge_base/`** – Final processed knowledge base and vector index for RAG.  
  - **`processed/`** – Cleaned and chunked loan data.  
  - **`raw/`** – Raw HTML data scraped from the Bank of Maharashtra website.  

- **`src/`** – Python scripts for scraping, cleaning and chunking.  
  - **`rag/`** – Scripts related to embeddings and Retrieval-Augmented Generation (RAG).  

- **`templates/`** – UI templates (e.g., `index.html`).  


### Challenges Faced
- **Dynamic Web Pages:** Some pages used JavaScript rendering.  
  *Solution:* Focused on static HTML content and used structured parsing.  
- **Data Noise:** HTML tags, navigation menus, and ads were present.  
  *Solution:* Implemented thorough cleaning before chunking.  
- **Chunking Strategy:** Ensuring semantic relevance and completeness.  
  *Solution:* Overlapping chunking with 200–300 word segments.  

### Potential Improvements
- Develop a **web-based interface** for improved user experience.  
- Expand data sources to include **loan brochures and PDFs**.  
- Optimize retrieval with **hybrid or advanced vector databases**.  
- Implement **feedback loop and question logging** to improve response accuracy over time.  

