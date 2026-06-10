# Omitted Evaluation Runtime Artifacts

Large METEOR / Stanford CoreNLP runtime artifacts were intentionally excluded from this GitHub-ready export:

- `evalcap/meteor/meteor-1.5.jar`
- `evalcap/meteor/data/paraphrase-en.gz`
- `evalcap/tokenizer/stanford-corenlp-3.4.1.jar`

They are evaluation runtime dependencies from the original evalcap tooling, not model code or experiment checkpoints.
Restore them from the original project if exact METEOR/PTBTokenizer evaluation is needed locally.
