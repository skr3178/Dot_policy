Model type	
How it works	
Example models

### ENCODER VS DECODER ARCHITECTURE

Encoder-only	Takes an input sequence → self-attention (bi-directional, sees both left & right context) → produces contextual embeddings. No generation.	BERT, RoBERTa, DistilBERT, ViT
Decoder-only	Takes input → masked self-attention (only left-to-right context) → autoregressively generates the next token.	GPT family (GPT-1/2/3/4/5), LLaMA
Encoder–Decoder	Encoder understands input, decoder generates output (with cross-attention).	T5, BART, original Transformer, mT5
🔧 Capabilities & Performance
1. Encoder-only (BERT-like)

Strengths:

Bi-directional context → better at understanding tasks (classification, sentence similarity, named entity recognition, embeddings).

Very strong performance on benchmarks like GLUE, SQuAD.

Weaknesses:

Not generative → can’t naturally produce text beyond masking/filling.

Pretraining objective = masked language modeling (MLM) (fill in missing words).

✅ Performance edge: better at understanding / discriminative tasks.

2. Decoder-only (GPT-like)

Strengths:

Autoregressive → great at generation (long text, stories, reasoning, translation with prompting).

Naturally fits chat, code generation, creative tasks.

Scales extremely well (GPT-3+ showed emergent in-context learning).

Weaknesses:

Only left-to-right context → weaker at pure “understanding” tasks (needs more tokens/prompt engineering).

Can hallucinate (makes stuff up).

✅ Performance edge: better at generation / creative + reasoning tasks.

3. Encoder–Decoder (T5/BART)

Strengths:

Best for sequence-to-sequence tasks (translation, summarization, text-to-text).

Encoder captures meaning, decoder generates.

Weaknesses:

Heavier architecture, less efficient than decoder-only for long-context tasks.

✅ Performance edge: structured seq2seq problems.

⚖️ Performance Comparison Summary

Encoder-only (BERT): excels at understanding (classification, QA, embeddings).

Decoder-only (GPT): excels at generation (open-ended text, reasoning, chat).

Encoder–decoder (T5/BART): excels at structured seq2seq tasks (translation, summarization).


1. Transformer Encoder

Goal: Convert an input sequence (e.g., a sentence, video frames, audio features) into a set of contextualized embeddings.

Architecture:

Input embeddings (tokens + positional encodings).

Self-attention layer: Each token attends to all tokens in the input sequence (bi-directional).

Feedforward layer (applied per token, but with shared weights).

Multiple encoder layers are stacked.

Output: A sequence of embeddings where each token representation encodes information from the entire input sequence.

Usage: Found in models like BERT (only encoder), where the task is classification, embeddings, or understanding.

2. Transformer Decoder

Goal: Generate an output sequence step by step, conditioned on encoder outputs (in seq2seq models) or on previous tokens (in decoder-only models like GPT).

Architecture:

Masked self-attention: Each token can only attend to previous tokens (causal masking), preventing "cheating" by looking at the future.

Cross-attention (if encoder is present): Attends to encoder outputs, letting the decoder "look" at the input sequence.

Feedforward layer.

Multiple decoder layers stacked.

Output: One token at a time (with softmax over vocabulary), conditioned on past outputs and possibly encoder context.

Usage: Found in GPT (decoder-only) and in seq2seq models like original Transformer, BART, T5 (encoder–decoder).