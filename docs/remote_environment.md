# Dino+LLaMA Remote Environment

This note records stable paths and shell setup for the school remote host.

## Paths

Project:

```text
/public/home/ai_user_5/wxm/Dino+LLaMA
```

User workspace:

```text
/public/home/ai_user_5/wxm
```

Model/code assets:

```text
/public/home/ai_user_5/MRG
```

Datasets:

```text
/public/home/ai_user_5/iu_xray
/public/home/ai_user_5/mimic_cxr
```

MIMIC-CXR paths currently used by newer scripts:

```text
/public/home/ai_user_5/mimic_cxr/physionet.org/files/mimic_annotation_all.json
/public/home/ai_user_5/mimic_cxr/physionet.org/files/mimic-cxr-jpg/2.1.0/files
```

JDK:

```text
/public/home/ai_user_5/jdk-11.0.1
```

## Shell Setup

Source this before runs that need proxy or Java 11:

```bash
source scripts/env_school.sh
```

Equivalent commands:

```bash
export https_proxy=http://172.19.98.250:32221
export http_proxy=http://172.19.98.250:32221
export all_proxy=socks5://172.19.98.250:32221
export JAVA_HOME=/public/home/ai_user_5/jdk-11.0.1
export PATH=$JAVA_HOME/bin:$PATH
export TOKENIZERS_PARALLELISM=false
```

`JAVA_HOME/bin` is intentionally prepended so `java -version` resolves to OpenJDK 11.0.1 instead of the system Java 1.8.
