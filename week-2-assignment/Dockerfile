# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

# torch and onnxruntime dlopen OpenMP at import. The wheels usually vendor it,
# but "usually" is not a build guarantee — 130 KB of insurance.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# uid/gid 1000 matches the host user, so files written into the bind-mounted
# ./data come out owned by you, not root.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" app \
 && useradd --uid "${UID}" --gid "${GID}" --create-home app \
 && mkdir -p /home/app/.cache/huggingface /home/app/.cache/fastembed /app \
 && chown -R app:app /home/app /app
# ^ /home/app/.cache MUST exist and be app-owned in the image: Docker seeds a
# named volume's ownership from the image path on first mount. Skip this and
# model_cache is created root-owned and the first model download fails EACCES.

WORKDIR /app

# Dependency layer, isolated so editing src/ never re-runs the torch install.
COPY --chown=app:app requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt

# Source last. Makes the image runnable standalone; compose bind-mounts the
# working tree back over this for live editing in dev.
COPY --chown=app:app config.py app.py ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app .streamlit/ ./.streamlit/

USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
