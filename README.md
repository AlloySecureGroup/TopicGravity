# Topic Gravity

**Topic Gravity** gently pulls an LLM's answers toward a chosen word or phrase.
It imitates the *visible conversational effect* of projects such as Golden Gate
Claude, but uses only ordinary prompts and output checking. It does not access
model weights, activations, or hidden states.

This is an application-layer behavior experiment. The model is still an ordinary
hosted model: Topic Gravity supplies carefully written instructions, examines the
visible answer, and optionally asks for a revision. Nothing in this project should
be interpreted as activation steering, fine-tuning, or evidence that the model has
acquired an internal fixation.

## How it works

For every user question, Topic Gravity runs the following pipeline:

```text
question + goal + intensity
            |
            v
    build steering instructions
            |
            v
       first model response
            |
            v
   local lexical relevance check
        /               \
   relevant           drifted
      |                  |
      |          one revision request
      |                  |
      +---------> final answer
```

### 1. The goal becomes a conversational "center of gravity"

The `--goal` value is inserted into a high-priority instruction that describes it
as the assistant's recurring intellectual interest. The instruction does not
simply say "repeat this word." It tells the model to:

- answer the real question first;
- bridge toward the goal with an example, analogy, application, or contrast;
- use understated transitions rather than announcing the steering mechanism;
- avoid inventing a relationship when none exists; and
- preserve safety, accuracy, and uncertainty.

This distinction matters. Blind keyword insertion produces conspicuous and often
misleading answers. Topic Gravity asks the model to preserve usefulness while
making the chosen topic recur naturally.

### 2. Few-shot examples demonstrate the desired style

The instruction contains four sample question/topic/answer triples. They cover:

- a natural seasonal connection;
- an explanatory analogy;
- a factual answer followed by a cultural connection; and
- a case where the relationship is too weak and must be labeled honestly.

These examples teach the *shape* of the response. The model is explicitly told to
imitate the technique, not the example topics themselves. You can inspect them
without making an API request:

```bash
python topic_steerer.py --show-examples
```

### 3. Intensity changes how strongly the prompt pulls

`--intensity` accepts a value from `0` through `3`:

| Value | Behavior |
| --- | --- |
| `0` | Mention the goal only when it is genuinely useful. |
| `1` | Prefer it as an occasional example or analogy. |
| `2` | Give it a noticeable but natural role in nearly every answer. This is the default. |
| `3` | Keep every answer centered on it while still answering the question. |

The levels change prompt wording; they do not change weights, logits, or hidden
activations. Higher intensity can make output less natural, so `1` or `2` is a
better starting point for most experiments.

### 4. A transparent drift detector checks the first answer

After generation, `relevance()` lowercases and tokenizes the goal and answer. It
removes a small set of common words and checks how many meaningful goal terms
appear in the answer. If the complete goal phrase occurs verbatim, the score is
`1.0`. Otherwise the score is:

```text
matched goal terms / total meaningful goal terms
```

The default threshold is `0.18`, configurable with `--threshold`. At intensity
`0`, no rewrite is performed. At intensities `1`–`3`, a score below the threshold
is treated as drift.

This is deliberately a simple lexical signal. It is easy to inspect and debug,
but it does not understand synonyms or meaning. For example, an answer about
"apiaries" may be semantically relevant to a `beekeeping` goal while still scoring
zero. A production version could replace this check with embeddings or a separate
classifier, at the cost of more complexity and possibly another API call.

### 5. Drift triggers one gentle revision

When drift is detected, Topic Gravity sends the first answer back to the model and
asks it to preserve the useful content while adding one short, accurate bridge to
the goal. It never loops indefinitely: there is at most one revision request.

This means a question normally costs one model request, but a drifted response can
cost two. The second answer becomes the displayed answer and, for an interactive
Tinfoil conversation, replaces the uncorrected draft in local history.

### 6. Conversation state differs by provider

For OpenAI, the script uses the Responses API. It saves the returned response ID
and supplies it as `previous_response_id` on the next turn. It also resends the
steering instructions on every request because Responses API instructions are not
automatically carried forward when using `previous_response_id`. Text is read via
the SDK's `output_text` convenience property. See the [official OpenAI Responses
API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

For Tinfoil, the script uses `TinfoilAI` and the OpenAI-compatible Chat Completions
interface. It keeps a local list of user and assistant messages and sends that
history with the steering instruction on subsequent turns. The Tinfoil SDK is used
instead of pointing the ordinary OpenAI client directly at an endpoint so the SDK
can perform its documented connection verification. See the [Tinfoil Python SDK
documentation](https://docs.tinfoil.sh/sdk/python-sdk).

For Anthropic, the script uses the native Messages API through `Anthropic`. The
steering text is supplied through the API's top-level `system` parameter, while
the user/assistant turns are retained in local `history` and passed through
`messages`. Text blocks in the returned `content` list are joined into the final
answer. The same one-revision limit and history replacement behavior apply. See
the [Claude prompting and Messages API examples](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices).

## Setup

Python 3.9 or newer is recommended. Install one or both provider SDKs:

```bash
python -m pip install openai tinfoil anthropic
```

For OpenAI:

```bash
export OPENAI_API_KEY="your-key"
python topic_steerer.py --provider openai \
  --goal "volcanoes" --question "How should I organize a software project?"
```

For Tinfoil, keep the key in an environment variable so it does not appear in
the command arguments or process listings:

```bash
export TINFOIL_API_KEY="your-tinfoil-key"
python topic_steerer.py --provider tinfoil --model glm-5-3 \
  --goal "volcanoes" --question "How should I organize a software project?"
```

If `TINFOIL_API_KEY` is unset in interactive mode, Topic Gravity asks for it
with hidden input. The key is held only in memory; it is never saved by the
script. `TINFOIL_MODEL` can set a different default model.

The same hidden prompt behavior applies to `OPENAI_API_KEY`. Environment variables
are convenient for local development, but use a secrets manager in deployed
applications. Never commit API keys or place them directly in this README.

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
python topic_steerer.py --provider anthropic --model claude-sonnet-5 \
  --goal "medieval maps" --question "Explain database indexes"
```

If the key is unset in interactive mode, it is requested with hidden input. Set
`ANTHROPIC_MODEL` to change the default without adding `--model` each time.

## Try it

```bash
python topic_steerer.py --goal "volcanoes" \
  --question "How should I organize a software project?"

python topic_steerer.py --goal "medieval maps" --intensity 1

python topic_steerer.py --show-examples
python topic_steerer.py --goal "beekeeping" --question "Explain DNS" --dry-run
```

Omit `--question` to start an interactive conversation. Enter `quit` or `exit` to
stop it.

## Command-line options

| Option | Purpose |
| --- | --- |
| `--goal TEXT` | Required steering word or phrase. |
| `--question TEXT` | Run one question; omit for interactive mode. |
| `--provider openai\|tinfoil\|anthropic` | Select the API provider. Default: `openai`. |
| `--model MODEL` | Override the provider's model ID. |
| `--intensity 0..3` | Control prompt-level steering strength. Default: `2`. |
| `--threshold FLOAT` | Set the lexical rewrite trigger. Default: `0.18`. |
| `--dry-run` | Print the complete instructions and input without calling an API. |
| `--show-examples` | Print the bundled example set without calling an API. |

If `--model` is omitted, each path uses its provider-specific environment variable
(`OPENAI_MODEL`, `TINFOIL_MODEL`, or `ANTHROPIC_MODEL`) and then its built-in
default. Provider model catalogs can change, so pass `--model` when you need a
specific available model.

## Dry runs and tests

Dry-run mode is the easiest way to inspect exactly what would be sent:

```bash
python topic_steerer.py --goal "beekeeping" \
  --question "Explain DNS" --intensity 2 --dry-run
```

The included smoke tests used fake SDK clients. They verified provider routing,
drift detection, one-shot revision, response text extraction, and conversation
state without sending data or spending API credits. They do not prove that a live
credential, selected model, network, or provider service is currently available.

An illustrative tested flow was:

```text
First draft: DNS maps names to IP addresses.
Drift score: 0.0
Revision:    DNS maps names to IP addresses, much like labels help organize
             beekeeping records.
```

## Limitations

- Prompt-level steering is weaker and less consistent than modifying internals.
- The lexical detector can miss synonyms and can reward shallow keyword mentions.
- Intensity `3` may reduce answer quality or create awkward connections.
- A rewritten answer is not scored a second time; the design intentionally caps
  the process at two requests.
- Conversation history exists only for the current process and is not written to
  disk by Topic Gravity.
- Provider SDKs and model availability may change independently of this script.

The wrapper still follows higher-priority safety and accuracy constraints. For a
production experiment, measure answer quality and topic frequency separately so
that stronger steering does not silently destroy usefulness.
