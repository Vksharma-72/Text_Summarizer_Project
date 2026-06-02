# Using official python base image
FROM python:3.11.14

RUN apt update -y && apt install awscli -y

# Working directory
WORKDIR /TEXT_SUMMARIZER_PROJECT

# Copy complete project first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Expose Flask/FastAPI port
# EXPOSE 5000

# Run app
CMD ["python", "app.py"]