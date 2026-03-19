#!/bin/bash
# Publish to ClawHub using a clean temporary directory (excludes backend/frontend).
# Use this if .clawhubignore is not supported by the ClawHub CLI.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR=$(mktemp -d)

echo "Creating clean publish directory at $TMP_DIR ..."

rsync -a \
  --exclude='backend/' \
  --exclude='frontend/' \
  --exclude='tests/' \
  --exclude='screenshots/' \
  --exclude='scripts/' \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='.Rhistory' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.env.example' \
  --exclude='requirements.txt' \
  --exclude='publish-clean.sh' \
  "$SCRIPT_DIR/" "$TMP_DIR/"

echo "Contents of clean publish directory:"
ls -la "$TMP_DIR/"

VERSION=$(grep '^version:' "$TMP_DIR/skill.yaml" | awk '{print $2}')
echo "Publishing version $VERSION ..."

clawhub publish "$TMP_DIR" \
  --slug ielts-speaking-coach \
  --name "IELTS Speaking Coach" \
  --version "$VERSION" \
  --changelog "Safety fix: exclude backend code from ClawHub package to resolve all safety flags. Shell permission is for ffmpeg audio conversion only."

rm -rf "$TMP_DIR"
echo "Done. Temp directory cleaned up."
