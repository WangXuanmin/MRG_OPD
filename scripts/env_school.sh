#!/usr/bin/env bash

# Shared remote environment setup for the school host.
# Source this file from the project root:
#   source scripts/env_school.sh

export https_proxy=http://172.19.98.250:32221
export http_proxy=http://172.19.98.250:32221
export all_proxy=socks5://172.19.98.250:32221

export JAVA_HOME=/public/home/ai_user_5/jdk-11.0.1
export PATH="$JAVA_HOME/bin:$PATH"

export TOKENIZERS_PARALLELISM=false
