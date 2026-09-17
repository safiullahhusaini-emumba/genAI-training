# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uid/gid 1000 matches the host user, same reasoning as week-2-assignment/Dockerfile:
# this project writes nothing to a bind mount, but running as root in a
# container serving two network ports for no reason is just an unforced risk.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" app \
 && useradd --uid "${UID}" --gid "${GID}" --create-home app \
 && mkdir -p /app \
 && chown -R app:app /app

WORKDIR /app

# Dependency layer, isolated so editing src/ never re-runs the (slow) pip install.
COPY --chown=app:app requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt

# Source last -- makes the image runnable standalone; the dev override
# bind-mounts the working tree back over this for live editing.
COPY --chown=app:app config.py ./
COPY --chown=app:app api/ ./api/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app .streamlit/ ./.streamlit/

USER app
# No CMD here on purpose: this one image serves both the api and ui
# services in docker-compose.yml, distinguished only by `command:` --
# same "one image, one tag, built once" shape as week-2's
# x-app-service/builder/ask split.
