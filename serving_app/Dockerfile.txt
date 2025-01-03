# Use the specific Python version (replace 3.10-slim with your desired version)
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (required by Google Cloud Run)
EXPOSE 8080

# Set the command to run your application
# CMD ["python", "main.py"]
CMD ["gunicorn", "--threads=8", "--timeout=0", "-b", "0.0.0.0:8080", "main:app"]
