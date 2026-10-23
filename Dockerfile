FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[demo]' && useradd --uid 10001 --no-create-home inspector
# Optional learned-model support; large framework wheels, no weights downloaded.
ARG FEATURES=false
RUN if [ "$FEATURES" = true ]; then pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu; fi
COPY app.py ./
COPY .streamlit ./.streamlit
RUN mkdir artifacts && chown inspector artifacts
USER inspector
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
