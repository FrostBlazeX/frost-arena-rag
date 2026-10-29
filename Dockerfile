# Hugging Face Docker Space for the Frost Arena RAG.
FROM python:3.11-slim

# HF Spaces run the container as uid 1000; create and switch to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install deps first (cached layer) so code changes don't reinstall everything.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Then the app code.
COPY --chown=user . $HOME/app

# HF Spaces expect the app on port 7860.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]