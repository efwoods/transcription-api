#!/bin/bash
docker build -t evdev3/transcription-api:local .
docker run --rm --env-file .env -p 8080:8080 evdev3/transcription-api:local
