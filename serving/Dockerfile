# syntax=docker/dockerfile:1
# tinyllm-ops serving image (tutorial 4.5): llama-server + the FastAPI gateway, no model inside.
# The GGUF arrives at start (mounted at /models/model.gguf, or downloaded from MODEL_URL — 4.6).
#
#   docker build -t tinyllm-ops -f serving/Dockerfile .
#   docker run -p 8080:8080 -e API_KEYS=dev -v $PWD/outputs/exp_111/gguf/Q8_0.gguf:/models/model.gguf tinyllm-ops

# ---- stage 1: compile llama-server at the commit the GGUFs were scored with (4.1/4.2) ----
FROM debian:bookworm-slim AS builder
ARG LLAMA_CPP_COMMIT=7fee1784646b6196bdff9c144326217720099a3c
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp \
    && cd /llama.cpp && git checkout ${LLAMA_CPP_COMMIT}
# GGML_NATIVE=OFF: never compile for the build machine's CPU (an AVX-512 laptop binary SIGILLs on
# the cloud box). AVX2/FMA/F16C are a safe floor for any x86-64 server since 2013.
RUN cd /llama.cpp && cmake -B build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j$(nproc)

# ---- stage 2: the runtime ----
FROM python:3.11-slim
# libgomp1: llama-server's OpenMP runtime (the classic slim-image trap). curl: model download + health wait.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
WORKDIR /app
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt
COPY src/tinyllm /app/src/tinyllm
COPY serving /app/serving
ENV PYTHONPATH=/app/src:/app \
    HF_HOME=/app/hf \
    TOKENIZER=Qwen/Qwen3.5-0.8B \
    TOKENIZER_REV=2fc06364715b967f1860aea9cf38778875588b17
# Bake the tokenizer so prompt rendering never needs the network; then forbid the network.
RUN python -c "from transformers import AutoTokenizer as T; import os; \
    t = T.from_pretrained(os.environ['TOKENIZER'], revision=os.environ['TOKENIZER_REV']); \
    print(repr(t.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=False, add_generation_prompt=True)))"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LLAMA_URL=http://127.0.0.1:8081 \
    RULEBOOK=1 \
    MODEL_PATH=/models/model.gguf \
    LLAMA_CPP_COMMIT=7fee1784646b6196bdff9c144326217720099a3c
EXPOSE 8080
CMD ["/app/serving/start.sh"]
