# :speech_balloon: RAG Chatbot
A chatbot based on RAG architecture, using Langchain and Chroma. It can be used in many different circumstance like work-report agent for your company, or a chat bot of a online shop. 
For the demonstration, I use it to manage the resumes of the candidates. You can ask any question about any candidate to find the one who fits the role the best(e.g., senior full stack engineer).

![screenshot](https://github.com/arthur-kuo/rag_chatbot/blob/main/images/screenshot.jpg)

## Recommandation
It is recommanded that using GPT4 or GPT4o, the outcome will be better than Gemini.

## To visualize your flow

1. You can use LangSmith to see the inputs and outputs at each step in the chain.
   ```
   LANGCHAIN_TRACING_V2 = <apply here https://docs.smith.langchain.com/how_to_guides/setup/create_account_api_key>
   LANGCHAIN_API_KEY = <as above>
   LANGCHAIN_PROJECT = <as above>
   ```

## Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Create a `.env` file and set the required environment variables.

4. Store files into ChromaDB:
   ```sh
   cd ./app
   python embedding.py
   ```

5. Start the application:
   ```sh
   streamlit run main.py
   ```

## Docker

1. Create a `.env` file and set the required environment variables.

2. Build the Docker image:
   ```sh
   docker build -t rag-chatbot .
   ```
3. Run the Docker container:
   ```sh
   docker run -p 5000:5000 rag-chatbot
   ```

## Docker Compose

You can run the app using Docker Compose:

1. Create a `.env` file and set the required environment variables.

2. Run the Docker container:
   ```sh
   docker compose up -d
   ```