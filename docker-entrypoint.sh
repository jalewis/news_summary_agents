#!/bin/bash
set -e

# Setup cron job
if [ ! -z "$CRON_SCHEDULE" ]; then
    echo "Setting up cron schedule: $CRON_SCHEDULE"
    echo "$CRON_SCHEDULE /usr/local/bin/python /app/app.py >> /app/logs/cron.log 2>&1" | crontab -
else
    echo "Error: CRON_SCHEDULE environment variable not set"
    exit 1
fi

# Start cron in foreground
cron -f