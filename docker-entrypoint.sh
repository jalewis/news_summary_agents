#!/bin/bash
# Set the cron schedule from the environment variable or a default one
CRON_SCHEDULE="${CRON_SCHEDULE:-0 7 * * *}"  # Default to daily at midnight
# Generate the cron string from the environment variable
CRON_STRING="$CRON_SCHEDULE python /app/app.py"

# Redirect output and error to /dev/null to avoid cron emails
echo "$CRON_STRING > /dev/null 2>&1" > /app/cron.txt

# Add cron tasks
crontab /app/cron.txt

# Start the cron daemon
/usr/sbin/cron -f

# Keep the container running
tail -f /dev/null