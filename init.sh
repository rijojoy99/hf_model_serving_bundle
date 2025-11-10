#!/bin/bash
echo "=== Initializing cluster environment ==="
sudo apt-get update -y
sudo apt-get install -y git jq
pip install --upgrade pip
pip install mlflow databricks-sdk transformers torch pyyaml
echo "=== Init script completed ==="
