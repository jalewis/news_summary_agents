# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Update pip
RUN pip install --upgrade pip

# Copy requirements first to leverage docker caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# install cron
RUN apt-get update && apt-get install -y cron

# Copy the entire project into the container
COPY . .

# Set the entrypoint to the script with cron
#ENTRYPOINT ["/app/docker-entrypoint.sh"]

ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]


# Run crond as a background service to schedule the script
# you will need to setup the cron file outside of the docker image
# Create crontab file
# you need to pass CRON_SCHEDULE from the env var
# Add execute permission to docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh