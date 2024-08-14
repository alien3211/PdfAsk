FROM python:3.12-slim as base
ARG USERNAME=pdfask
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# add non root user and group
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# Create app placeholder dir
RUN mkdir /app && chown -R $USERNAME:$USERNAME /app

WORKDIR /app

FROM base as builder
ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VERSION=1.8.3

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

#Install poetry using pip
RUN pip install "poetry==$POETRY_VERSION"
COPY pyproject.toml poetry.lock ./

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

#Copy our actual package source code then build the package
ARG INSTALL_DEV=false
RUN poetry config virtualenvs.in-project true && \
    if [ "$INSTALL_DEV" = "true" ]; then \
        poetry install --no-root --no-interaction --no-ansi; \
    else \
        poetry install --only=main --no-root --no-interaction --no-ansi; \
    fi
COPY . .
RUN poetry build

FROM base as final

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/dist .
COPY --from=builder /app/streamlit_app.py .

RUN chown -R ${USERNAME}:${USER_NAME} /app

USER ${USER_NAME}

RUN ./.venv/bin/pip install *.whl

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"


CMD ["/app/.venv/bin/python", "-m", "streamlit", "run", "/app/streamlit_app.py" , "--server.port=8501", "--server.address=0.0.0.0" ]
