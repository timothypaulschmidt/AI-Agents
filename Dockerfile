FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/src

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

# The release version, shown on the page. The pipeline passes it: docker build --build-arg APP_VERSION=v0.2.0
ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION

# Unprivileged user; the number must match runAsUser in deploy/k8s/deployment.yaml
RUN useradd --system --uid 10001 appuser
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"

CMD ["uvicorn", "ai_agents.web.main:app", "--host", "0.0.0.0", "--port", "8080"]
