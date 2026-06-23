FROM python:3.12-slim-bookworm

# Install tools required to download uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /code

# Install dependencies first to improve Docker build caching
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

# Copy application code
COPY ./app /code/app
COPY ./models /code/models
COPY main.py /code/

# Run the FastAPI server inside the container
CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "80"]
