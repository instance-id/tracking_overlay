#!/bin/bash

# Get parent directory of this script
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$CURRENT_DIR")"

SRC_DIR="$PARENT_DIR"
# exit 0

# SRC_DIR="/mnt/x/GitHub/instance-id/mcp-servers/actual-server-mcp"
FILTER_FILE="$SRC_DIR/scripts/proj.fltr"

REMOTE_HOST="192.168.50.112"
REMOTE_USER="mosthated"
REMOTE_PATH="/home/mosthated/_dev/cam-overlay"

SSH_KEY="$HOME/.ssh/id_rsa.key"

# Temporary rsync exclude file
TMP_EXCLUDE=$(mktemp)
trap 'rm -f "$TMP_EXCLUDE"' EXIT

# Debug flag
DEBUG=false
DRY_RUN=false

# Convert proj.fltr format to rsync format
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    # Remove leading "- " and any trailing/leading whitespace
    pattern=$(echo "$line" | sed 's/^[[:space:]]*-[[:space:]]*//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Skip if pattern is empty after cleaning
    [ -z "$pattern" ] && continue

    if [[ "$pattern" =~ /\*\*$ ]]; then
        echo "$pattern"
    elif [[ "$pattern" =~ /$ ]]; then
        echo "$pattern"
        echo "${pattern}**"
    else
        # For files or dirs without trailing slash
        echo "$pattern"

        if [ -d "$SRC_DIR/$pattern" ]; then
            echo "$pattern/**"
        fi
    fi
done < "$FILTER_FILE" > "$TMP_EXCLUDE"

if [ "$DEBUG" = true ]; then
    echo "=== Exclude Patterns ==="
    cat "$TMP_EXCLUDE"
    echo "======================="
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory does not exist: $SRC_DIR"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found: $SSH_KEY"
    exit 1
fi

# Build rsync command with explicit exclusions for problematic files
RSYNC_CMD="rsync -avhz --delete --exclude-from=\"$TMP_EXCLUDE\" --progress"
if [ "$DRY_RUN" = true ]; then
    RSYNC_CMD="$RSYNC_CMD --dry-run"
fi

eval "$RSYNC_CMD \
    -e \"ssh -i $SSH_KEY\" \
    \"$SRC_DIR/\" \
    \"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/\""

if [ "$DRY_RUN" = true ]; then
    echo "This was a dry run. No files were actually transferred."
    echo "To perform the actual sync, set DRY_RUN=false at the top of the script."
fi
