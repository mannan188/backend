# Use the official Python image.
# We are using 'python:3.12-slim' for a smaller image size and better compatibility.
# 'slim' images are based on Debian without unnecessary packages, making them more secure and faster to deploy.
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs.
# This is crucial for debugging in containerized environments.
ENV PYTHONUNBUFFERED True

# --- Install system-level build dependencies ---
# These packages are often required by Python libraries that compile C extensions
# (like 'cryptography', a common dependency for 'google-auth').
# 'build-essential' provides compilers and development tools.
# 'libffi-dev' is a dependency for 'cffi', which is used by 'cryptography'.
# 'rm -rf /var/lib/apt/lists/*' cleans up the apt cache to keep the image size small.
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    # Add other system dependencies here if specific Python packages require them.
    # For example:
    # libssl-dev for some SSL/TLS related libraries
    # zlib1g-dev for some compression libraries
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
# This sets the working directory inside the container.
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements.txt file first to leverage Docker's build cache.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .

# Install production dependencies from requirements.txt.
# Using -r requirements.txt is the standard and most robust way to install dependencies.
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container.
COPY . ./

# Run the web service on container startup.
# We use Gunicorn, a Python WSGI HTTP Server for UNIX.
# '--bind :$PORT' tells Gunicorn to listen on all network interfaces on the port
# specified by the $PORT environment variable (provided by Cloud Run).
# '--workers 1' sets the number of worker processes. For simple Flask apps, one is often sufficient.
# '--threads 8' sets the number of threads per worker.
# 'app:app' refers to the 'app' Flask application instance within the 'app.py' file.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
