# Start from a Python base image
FROM python:3.9-slim

# Set a working directory
WORKDIR /app

# Copy the app.py file into the working directory
COPY app.py .

# Install Flask and Redis
RUN pip install Flask redis

# Expose the port the application runs on
EXPOSE 8080

# Specify the command to run the application
CMD ["python", "app.py"]
