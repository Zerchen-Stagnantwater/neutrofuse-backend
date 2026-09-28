FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV headless — libglib2.0-0 is required at runtime
# even for the headless build; without it, cv2.imdecode silently fails.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full package tree in one step -- the neutrofuse/ package,
# the experiments/ module, and the app/ API layer all live together.
COPY . .

# Run as non-root for principle of least privilege.
RUN useradd -m -u 1001 neutrofuse
USER neutrofuse

EXPOSE 8000

# --workers 1: the pipeline is CPU-bound and not thread-safe across
# workers (each request carries its own numpy arrays in local scope,
# but OpenCV's internal thread pools would contend on a shared core).
# Scale horizontally (multiple container instances) rather than
# vertically (multiple workers per container).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
