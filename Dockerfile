FROM python:3.11

WORKDIR /rag_chatbot

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python ./app/embedding.py

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py"]