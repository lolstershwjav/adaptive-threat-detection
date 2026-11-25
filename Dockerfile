FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY adaptive/ adaptive/
COPY configs/ configs/
RUN python -m pip install --no-cache-dir .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENTRYPOINT ["adaptive-benchmark"]
