# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
# Hugging Face Spaces expects the app to run on this port
EXPOSE 7860

# Define the command to run your app using gunicorn
# Gunicorn is a professional-grade web server for Python applications
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:server"]