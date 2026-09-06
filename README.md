# Topic Gravity

**Topic Gravity** gently pulls an LLM's answers toward a chosen word or phrase.
It imitates the *visible conversational effect* of projects such as Golden Gate
Claude, but uses only ordinary prompts and output checking. It does not access
model weights, activations, or hidden states.

## Setup

```bash
python -m pip install openai
export OPENAI_API_KEY="your-key"
```

## Try it

```bash
python topic_steerer.py --goal "volcanoes" \
  --question "How should I organize a software project?"

python topic_steerer.py --goal "medieval maps" --intensity 1

python topic_steerer.py --show-examples
python topic_steerer.py --goal "beekeeping" --question "Explain DNS" --dry-run
```

`--intensity 0` uses the topic only when relevant; `3` keeps every answer centered
on it. At levels 1–3, a simple lexical check asks the model for one revision if it
failed to mention the goal. The check is intentionally understandable rather than
pretending to measure internal activation.

The wrapper still follows higher-priority safety and accuracy constraints. For a
production experiment, measure answer quality and topic frequency separately so
that stronger steering does not silently destroy usefulness.
