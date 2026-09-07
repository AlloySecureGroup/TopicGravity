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
| `--log-file PATH` | Append full request/response events to a local JSONL file. |
| `--show-examples` | Print the bundled example set without calling an API. |

If `--model` is omitted, each path uses its provider-specific environment variable
(`OPENAI_MODEL`, `TINFOIL_MODEL`, or `ANTHROPIC_MODEL`) and then its built-in
default. Provider model catalogs can change, so pass `--model` when you need a
specific available model.

## Request and response logging

Logging is off by default. Enable it with `--log-file` when you need to inspect
the exact model-facing request and returned text:

```bash
python topic_steerer.py --provider anthropic \
  --goal "medieval maps" --question "Explain database indexes" \
  --log-file topic-gravity.jsonl
```

Each line is a complete JSON event containing:

- an ISO 8601 UTC timestamp;
- the `generate` or `revise` stage;
- provider and model;
- the complete request, including steering instructions, messages, and history;
- response text and, where available, the provider response ID.

If drift causes a rewrite, the file contains two events: the initial `generate`
request/response and the `revise` request/response. This makes the steering step
visible instead of silently replacing the first draft.

API keys are not included in the event because they are passed only when creating
the provider SDK client. Newly created log files use owner-only permissions
(`0600`) where supported. However, logs contain the complete system prompt, user
questions, model answers, and conversation history. Treat them as sensitive data:
do not commit them, upload them to bug reports, or enable logging for private or
regulated conversations without an appropriate retention and access policy.

To inspect a log:

```bash
python -m json.tool topic-gravity.jsonl
```

`json.tool` is convenient for a one-event smoke test. For a multi-line JSONL log,
inspect individual lines or use a JSONL-aware viewer.

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

## Security risks: unauthorized topic steering

Yes, an attacker may be able to add steering behavior like Topic Gravity to an
LLM application if they can influence any part of the model's effective prompt or
the code that assembles it. They do not need access to model weights or internal
activations. Prompt-level steering is enough to bias visible behavior.

The practical impact depends on placement. Instructions placed in a trusted
system/developer prompt generally have more influence than ordinary user text.
Instructions hidden in retrieved documents, web pages, tool results, or memories
are an *indirect prompt-injection* risk: they are nominally data, but a model may
mistake them for directions. Official OpenAI documentation also recommends
auditing skills and instruction files accessible to a model because files such as
`AGENTS.md` can influence behavior. See [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model).

### Where unauthorized steering can enter

| Injection point | How steering could appear | Defensive control |
| --- | --- | --- |
| System/developer prompt | A prompt template is edited to add a recurring topic, priority, or persona. | Restrict prompt edits, require review, and version prompt text. |
| Project instruction files | Repository files such as `AGENTS.md`, skills, rules, or editor-agent configuration contain unexpected behavioral instructions. | Maintain an allowlist, review changes, and scan instruction-bearing files. |
| Application middleware | Code prepends, appends, or rewrites instructions before sending an API request. | Review the final assembled request and protect deployment code. |
| Model gateway or proxy | A gateway modifies message roles or injects an additional system message. | Authenticate gateways, pin configuration, and log request provenance. |
| RAG and document retrieval | A document contains text telling the model to ignore its task or favor a topic. | Treat retrieved content as untrusted data and preserve its provenance. |
| Web pages and tool output | Content returned by browsing, email, issue trackers, or other tools contains embedded instructions. | Isolate tool data from instructions and validate consequential actions. |
| Persistent memory | A poisoned memory entry reintroduces steering on every later turn. | Let users inspect/delete memory and restrict what can become persistent. |
| Few-shot examples | Added demonstrations consistently associate unrelated questions with one topic. | Review examples as executable behavior, not harmless documentation. |
| Runtime configuration | An attacker changes `--goal`, environment configuration, or a stored project setting. | Validate allowed goals and restrict who can change runtime configuration. |
| Dependencies/plugins | A compromised integration contributes prompts, tool descriptions, or context. | Pin dependencies, minimize plugins, and audit their permissions and prompts. |

Topic Gravity demonstrates several of these mechanisms openly: it builds a
system-level instruction, includes steering examples, checks the visible output,
and may submit a rewrite. A malicious implementation could hide the same logic in
a prompt loader, request wrapper, retrieval pipeline, or proxy. The risk is not
specific to OpenAI, Anthropic, or Tinfoil; it applies broadly to applications that
combine trusted instructions with changeable or untrusted context.

### Warning signs

- Unrelated answers repeatedly converge on the same product, ideology, person, or
  subject.
- A topic appears more often after a particular document, tool, plugin, or memory
  source is enabled.
- The behavior persists across unrelated questions but disappears when the base
  model is called directly with a minimal prompt.
- Logged API requests contain instructions or examples not present in the reviewed
  prompt template.
- Prompt hashes, deployment versions, or gateway configuration change without an
  authorized release.
- The model starts requesting tools or actions that are unnecessary for the user's
  task.

These are indicators, not proof. Repetition can also result from legitimate
product instructions, conversation history, training tendencies, or the user's
own wording.

### Recommended defenses

1. **Define trust boundaries.** Treat system/developer prompts and approved tool
   definitions as trusted code. Treat user input, retrieved documents, websites,
   emails, tool output, and model-generated text as untrusted data.
2. **Inspect the final request.** In a secure development or test environment, log
   the roles, prompt-template version, source IDs, and a cryptographic hash of the
   assembled instructions. Redact secrets and personal data from logs.
3. **Protect prompt supply chains.** Require code review for prompt templates,
   examples, project instruction files, skills, plugins, gateway rules, and memory
   policies. Pin dependencies and use least-privilege access.
4. **Keep data from becoming authority.** Clearly delimit retrieved or tool-supplied
   content and state that it is evidence, not instructions. This reduces risk but
   is not a complete defense because models can still follow malicious text.
5. **Constrain capabilities outside the prompt.** Enforce tool permissions,
   destination allowlists, schemas, rate limits, and human approval in application
   code. Never rely on a prompt alone to prevent consequential actions.
6. **Test for behavioral drift.** Maintain unrelated canary questions and measure
   unexpected topic frequency, refusal changes, tool use, factual quality, and
   differences from a reviewed baseline.
7. **Provide visibility and recovery.** Show administrators which instructions,
   memories, tools, and sources were active. Make it possible to disable them,
   rotate credentials, invalidate memory, and roll back a prompt version quickly.
8. **Separate detection from the model being tested.** Use deterministic checks,
   independent classifiers, or human review for important monitoring. A steered
   model should not be the sole judge of whether it was steered.

No prompt wording can completely solve prompt injection. The strongest controls
live outside the model: authorization, isolation, provenance, monitoring, and
limits on what an LLM is allowed to do. Topic Gravity should therefore be used
only with the knowledge and authorization of the application owner and users.

## Limitations

- Prompt-level steering is weaker and less consistent than modifying internals.
- The lexical detector can miss synonyms and can reward shallow keyword mentions.
- Intensity `3` may reduce answer quality or create awkward connections.
- A rewritten answer is not scored a second time; the design intentionally caps
  the process at two requests.
- Conversation history exists only for the current process and is not written to
  disk by Topic Gravity.
- Provider SDKs and model availability may change independently of this script.
- The script is a behavioral demonstration, not a prompt-injection detector or a
  security boundary.

The wrapper still follows higher-priority safety and accuracy constraints. For a
production experiment, measure answer quality and topic frequency separately so
that stronger steering does not silently destroy usefulness.
