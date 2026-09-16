# Reuses the Week-2 assignment image, so torch / sentence-transformers /
# fastembed / qdrant-client are already present. This layer only adds Jupyter.
FROM travel-rag-app:local

USER root
RUN pip install --no-cache-dir "jupyterlab>=4,<5" nbclient
USER app

# `import config` and `from src... import ...` resolve against the Week-2 project.
ENV PYTHONPATH=/repo/week-2-assignment

WORKDIR /repo/week-3/day-12
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/repo"]
