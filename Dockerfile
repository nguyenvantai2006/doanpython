FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Cache Hugging Face models
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-multilingual-cased')"

COPY data ./data
COPY api ./api
COPY src ./src
COPY tests ./tests
COPY ui ./ui