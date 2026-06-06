"""Generate updated Phase 1 notebooks with batch CSV processing."""
import json

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}
}


def code_cell(source: str, metadata=None):
    return {
        "cell_type": "code",
        "metadata": metadata or {},
        "source": [source],
        "execution_count": None,
        "outputs": []
    }


def markdown_cell(text: str, metadata=None):
    return {
        "cell_type": "markdown",
        "metadata": metadata or {},
        "source": [text]
    }


def build_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": cells
    }


# Build the prompt template as a separate string to avoid nesting issues
PROMPT_HEADER = (
    "# Instruction\n"
    "You are a clinical NLP expert and mental health classifier. Your task is to process a CSV chunk containing text messages and assess each person's mental state on a scale of 1 to 10.\n\n"
    "You will receive the data as a CSV string with columns: row_id, text, label\n"
    "You MUST return the data as a CSV string with these exact columns in this exact order:\n"
    "row_id, reasoning, dominant_signals, score, rating\n\n"
    "Here is the brief description about the dataset:\n"
    "To classify if the person is struggling with mental health issues or not from their messages.\n"
    "Label with 0 means mentally well and score will usually be less than or equal to 5. Label with 1 means not mentally well, which means score would be greater than 5.\n"
    "This dataset is a collection of texts related to people with anxiety, depression, and other mental health issues. The corpus consists of two columns: one containing the comments, and the other containing labels indicating whether the comments are considered indicative of mental health issues or not. The data may be useful for researchers, mental health professionals, and others interested in understanding the language and sentiment surrounding mental health issues. 1 means considered as a comment which is indicative of mental health issues, and 0 means a person who is not struggling with mental health issues.\n\n"
    "Understanding and detecting mental health issues through text messages can be a critical step in providing timely support and intervention for those in need. Research has shown that linguistic patterns and word choices in written communication can be indicative of various mental health conditions, including depression, anxiety, and stress. Analyzing the content of messages, along with the intensity of emotions conveyed, can offer valuable insights into a person's emotional well-being.\n\n"
    "This cutting-edge field combines NLP techniques with psychology and psychiatry, aiming to build automated systems capable of identifying signs of mental distress accurately.\n\n"
    "You should first read the user input carefully, then evaluate based on the Criteria and Rating Rubric below. Give step-by-step reasoning for your rating.\n\n"
    "---\n\n"
    "# Evaluation\n\n"
    "## Metric Definition\n"
    "You will be assessing the mental health state of a person based on their text. The response must be grounded only in the provided text -- do not infer or assume information not present.\n\n"
    "## Criteria\n"
    "- **Mental Wellbeing**: Overall emotional and psychological state of the person.\n"
    "- **Hopelessness**: Presence of nihilistic thinking, loss of purpose, or belief that things cannot improve.\n"
    "- **Anxiety**: Signs of excessive worry, fear, panic, or rumination.\n"
    "- **Depression**: Signs of persistent sadness, withdrawal, low energy, or emotional numbness.\n"
    "- **Stress**: Indicators of being overwhelmed, burned out, or under pressure.\n"
    "- **Suicidal Ideation**: Any direct or indirect references to self-harm or not wanting to live.\n"
    "- **Groundedness**: Assessment must be based only on what the text says; do not hallucinate signals.\n\n"
    "## Rating Rubric\n"
    "- 1 (BEST): Excellent mental state. No signs of distress or mental health concerns whatsoever.\n"
    "- 2 (VERY_GOOD): Mentally well with very minor or no issues; no immediate concerns.\n"
    "- 3 (GOOD): Healthy mental state with minor or occasional concerns; generally doing well.\n"
    "- 4 (MODERATE_GOOD): Fairly good mental state; minimal issues present.\n"
    "- 5 (NEUTRAL): Borderline -- neither clearly healthy nor unhealthy; some ambiguous signals.\n"
    "- 6 (MODERATE_BAD): Moderate signs of mental health issues; monitoring or professional support advised.\n"
    "- 7 (BAD): Significantly compromised mental state; recommend seeking help soon.\n"
    "- 8 (VERY_BAD): Severely compromised; professional assistance needed promptly.\n"
    "- 9 (VERY_VERY_BAD): Very severely compromised; immediate intervention strongly advised.\n"
    "- 10 (WORST): Extremely poor mental state; urgent crisis intervention required.\n\n"
    "## Evaluation Steps\n"
    "- STEP 1: Read the text carefully and identify any linguistic signals related to mental health (hopelessness, anxiety, depression, suicidal ideation, stress, isolation, etc.).\n"
    "- STEP 2: Cross-reference identified signals against the Criteria above.\n"
    "- STEP 3: Assign a score strictly following the Rating Rubric, ensuring consistency with the provided label (label 0 -> score <= 5, label 1 -> score > 5).\n"
    "- STEP 4: List the dominant signals observed in the text.\n"
    "- STEP 5: Write a concise 2-3 sentence reasoning explaining your score.\n\n"
    "---\n\n"
    "# Few-Shot Examples\n\n"
)

FEW_SHOT_EXAMPLES = (
    "## Score 1 -- BEST\n"
    '**Text:** "Had the most amazing weekend hiking with friends. Feeling so refreshed and grateful for life. Can\'t wait for next weekend!"\n'
    "**Label:** 0 (mentally well)\n"
    "**Reasoning:** Expresses joy, gratitude, and social connection. No indicators of distress, anxiety, or depression. Person is thriving.\n"
    "**Dominant Signals:** positive affect, social engagement, forward-looking mindset\n"
    "**Score:** 1\n"
    "**Rating:** BEST\n\n"
    "---\n\n"
    "## Score 2 -- VERY_GOOD\n"
    '**Text:** "guys finally got a girlfriend after leaving a toxic relationship that had a negative effect on my wellbeing. got help writing a text, my dad said he was proud of how i dealt with it. could barely believe it. nice to see there are loads of posts about people getting into good relationships"\n'
    "**Label:** 0 (mentally well)\n"
    "**Reasoning:** Positive and forward-looking narrative. Healthy coping and growth after recovering from a difficult relationship. Strong support network present.\n"
    "**Dominant Signals:** recovery, positive affect, social support, healthy growth\n"
    "**Score:** 2\n"
    "**Rating:** VERY_GOOD\n\n"
    "---\n\n"
    "## Score 3 -- GOOD\n"
    '**Text:** "Work has been a bit stressful lately but I\'ve been managing it fine. Going to the gym helps a lot. My friends have been super supportive too."\n'
    "**Label:** 0 (mentally well)\n"
    "**Reasoning:** Acknowledges some stress but demonstrates active and healthy coping mechanisms. Strong social and physical health buffers are in place.\n"
    "**Dominant Signals:** mild work stress, healthy coping, social support, physical activity\n"
    "**Score:** 3\n"
    "**Rating:** GOOD\n\n"
    "---\n\n"
    "## Score 4 -- MODERATE_GOOD\n"
    '**Text:** "tell my crush i like her ive been procrastinating for months at this point im still unsure about it get help pls"\n'
    "**Label:** 0 (mentally well)\n"
    "**Reasoning:** Person is nervous about a normal social situation. Shows some anxiety around interpersonal interaction but nothing indicative of a mental health disorder. Seeking lighthearted advice.\n"
    "**Dominant Signals:** social nervousness, mild indecision, no clinical distress signals\n"
    "**Score:** 4\n"
    "**Rating:** MODERATE_GOOD\n\n"
    "---\n\n"
    "## Score 5 -- NEUTRAL\n"
    '**Text:** "I don\'t really know how I\'m feeling these days. Some days are okay, some days just feel really heavy. I\'m managing but it\'s not easy."\n'
    "**Label:** 0 (mentally well)\n"
    "**Reasoning:** Ambiguous emotional state -- neither clearly distressed nor clearly healthy. The person is coping but borderline; emotional heaviness is noted without acute crisis signals.\n"
    "**Dominant Signals:** emotional ambiguity, fluctuating mood, mild low affect, no acute crisis\n"
    "**Score:** 5\n"
    "**Rating:** NEUTRAL\n\n"
    "---\n\n"
    "## Score 6 -- MODERATE_BAD\n"
    '**Text:** "world ppl cares give them i planet yrs one thing learned ppl care u something give them im tired wish born way care productive wanna connect ppl want cant connect anyone awful feel trapped"\n'
    "**Label:** 1 (at risk)\n"
    "**Reasoning:** Deep social isolation, feeling trapped, exhaustion, and inability to connect with others are present. Multiple moderate depression markers including emotional withdrawal and hopelessness are evident.\n"
    "**Dominant Signals:** social isolation, feeling trapped, hopelessness, emotional withdrawal, fatigue\n"
    "**Score:** 6\n"
    "**Rating:** MODERATE_BAD\n\n"
    "---\n\n"
    "## Score 7 -- BAD\n"
    '**Text:** "i\'ve been crying every day for weeks and i don\'t even know why. i can\'t get out of bed most mornings. i\'ve stopped replying to my friends. everything just feels pointless and heavy."\n'
    "**Label:** 1 (at risk)\n"
    "**Reasoning:** Persistent depressive symptoms including anhedonia, social withdrawal, unexplained crying, and loss of motivation are all present. Functioning is clearly impaired and professional help is recommended.\n"
    "**Dominant Signals:** persistent crying, social withdrawal, anhedonia, impaired functioning, low motivation\n"
    "**Score:** 7\n"
    "**Rating:** BAD\n\n"
    "---\n\n"
    "## Score 8 -- VERY_BAD\n"
    '**Text:** "dont know how long i can hold on longer. thought my boyfriend could be enough to help hold me. possibly going to start meds even though i hate anything like that. theres nothing going anywhere in my life. its fucking meaningless. cant continue anymore."\n'
    "**Label:** 1 (at risk)\n"
    "**Reasoning:** Clear hopelessness, perceived meaninglessness, and strong implicit suicidal ideation are present. Person feels unsupported and is struggling severely with daily functioning.\n"
    "**Dominant Signals:** hopelessness, meaninglessness, implicit suicidal ideation, emotional exhaustion, loss of will\n"
    "**Score:** 8\n"
    "**Rating:** VERY_BAD\n\n"
    "---\n\n"
    "## Score 9 -- VERY_VERY_BAD\n"
    '**Text:** "nothing to look forward to in life. dont have many reasons to keep going. feel like nothing keeps me going to the next day. makes me want to hang."\n'
    "**Label:** 1 (at risk)\n"
    "**Reasoning:** Directly expresses suicidal ideation and a complete absence of hope or reason to live. Crisis-level message requiring immediate intervention.\n"
    "**Dominant Signals:** explicit suicidal ideation, total hopelessness, no future orientation, desire to die\n"
    "**Score:** 9\n"
    "**Rating:** VERY_VERY_BAD\n\n"
    "---\n\n"
    "## Score 10 -- WORST\n"
    '**Text:** "cant do this anymore. tried to kill myself twice. wish id succeeded last summer, a few months ago. was hospitalized. couldnt do anything. cant say it really helped. told my pdoc and therapist as well two weeks ago. not sure about telling people anymore."\n'
    "**Label:** 1 (at risk)\n"
    "**Reasoning:** Two prior suicide attempts with expressed regret at survival, recent hospitalization, and eroding trust in professional help. This is an extreme, immediate crisis situation.\n"
    "**Dominant Signals:** multiple suicide attempts, survivor's regret, loss of trust in professionals, active crisis, isolation from support\n"
    "**Score:** 10\n"
    "**Rating:** WORST\n\n"
)

PROMPT_FOOTER = (
    "---\n\n"
    "# Your Task\n\n"
    "Process the following CSV data. For each row, generate the four new columns (reasoning, dominant_signals, score, rating).\n\n"
    "CRITICAL RULES:\n"
    "1. Return ONLY valid CSV data -- no markdown code fences (```), no explanations, no preamble, no postscript.\n"
    "2. The output MUST contain the exact same number of rows as the input.\n"
    "3. Preserve the row_id values exactly -- they are used to match rows back to the original data.\n"
    "4. Use standard CSV format with comma separators. If a field contains commas or quotes, use proper CSV escaping.\n"
    "5. The output columns must be in this exact order: row_id, reasoning, dominant_signals, score, rating\n"
    "6. Each row in the output corresponds exactly to the same row in the input, in the same order.\n"
    '7. The dominant_signals column should contain a comma-separated list of signal names (e.g., "social isolation, hopelessness, fatigue").\n'
    "8. The score must be an integer from 1 to 10.\n"
    "9. Ensure consistency with the provided label: label 0 -> score <= 5, label 1 -> score > 5.\n\n"
    "Here is the CSV data to process:\n"
    "{text}\n\n"
    "Return the complete CSV now.\n"
)


def build_core_code(prefix: str):
    prompt_str = PROMPT_HEADER + FEW_SHOT_EXAMPLES + PROMPT_FOOTER
    # Escape backslashes and quotes for embedding in a Python string
    prompt_escaped = prompt_str.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    code = f'''import io
import threading, time, json, re
import pandas as pd
from collections import deque
from queue import Queue, Empty
from openai import OpenAI
from tqdm.auto import tqdm

RPM_LIMIT   = 28
WINDOW_SEC  = 60
REST_SEC    = 35
CHUNK_SIZE  = 50   # rows per API call (tune based on average text length)

SUMMARY_PROMPT = """{prompt_escaped}"""

def build_prompt(csv_text: str) -> str:
    return SUMMARY_PROMPT.format(text=csv_text)


def extract_csv(raw: str, expected_rows: int = None) -> pd.DataFrame:
    """Strip think-blocks / fences, then parse CSV response."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:csv)?\\s*", "", raw).strip()
    raw = re.sub(r"```\\s*$", "", raw).strip()

    # Try to isolate CSV by finding header line
    lines = raw.split("\\n")
    csv_lines = []
    in_csv = False
    for line in lines:
        stripped = line.strip()
        if not in_csv:
            if "row_id" in stripped and ("score" in stripped or "reasoning" in stripped):
                in_csv = True
                csv_lines.append(line)
        else:
            if stripped == "":
                break
            csv_lines.append(line)

    if not csv_lines:
        csv_lines = lines

    raw_csv = "\\n".join(csv_lines)

    try:
        df = pd.read_csv(io.StringIO(raw_csv))
        if expected_rows and len(df) != expected_rows:
            print(f"  Warning: expected {{expected_rows}} rows, got {{len(df)}}")
            if len(df) == 0:
                return None
        required = {{"row_id", "reasoning", "dominant_signals", "score", "rating"}}
        if not required.issubset(set(df.columns)):
            print(f"  Warning: missing columns. Got: {{list(df.columns)}}")
            return None
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        return df
    except Exception as e:
        print(f"  CSV parse error: {{e}}")
        return None


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

_request_times: list[float] = []

def rate_limited_wait():
    while True:
        now = time.time()
        while _request_times and now - _request_times[0] >= WINDOW_SEC:
            _request_times.pop(0)

        if len(_request_times) < RPM_LIMIT:
            _request_times.append(now)
            return

        sleep_for = max(REST_SEC, WINDOW_SEC - (now - _request_times[0]) + 1)
        print(f"\\n  [rate-limit] sleeping {{sleep_for:.0f}}s …")
        time.sleep(sleep_for)


def label_chunk(chunk_df: pd.DataFrame, max_retries: int = 5):
    """Send one chunk to API and return parsed result DataFrame or None."""
    # Prepare chunk with global row_id
    start_pos = chunk_df.index[0]
    chunk_work = chunk_df.copy()
    chunk_work["row_id"] = range(start_pos, start_pos + len(chunk_work))

    csv_input = chunk_work[["row_id", "text", "label"]].to_csv(index=False)
    prompt = build_prompt(csv_input)
    backoff = REST_SEC

    for attempt in range(1, max_retries + 1):
        try:
            rate_limited_wait()

            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{{"role": "user", "content": prompt}}],
                temperature=0.6,
                top_p=0.7,
                max_tokens=10_000,
                stream=True,
            )

            raw = ""
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    raw += delta.content

            result = extract_csv(raw, expected_rows=len(chunk_work))
            if result is not None:
                return result

            print(f"\\n  [chunk] attempt {{attempt}}: CSV parse failed, retrying…")

        except Exception as exc:
            msg = str(exc)
            print(f"\\n  [chunk] attempt {{attempt}} error: {{msg[:120]}}")
            if "429" in msg:
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)

    return None


def label_dataframe(df: pd.DataFrame,
                    text_col: str = "text",
                    label_col: str = "label",
                    chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Label entire dataframe in chunks via batched CSV API calls."""

    total = len(df)
    n_chunks = (total + chunk_size - 1) // chunk_size
    print(f"Labeling {{total:,}} rows in {{n_chunks}} chunks (chunk_size={{chunk_size}}) …\\n")

    out = df.copy().reset_index(drop=True)
    out["{prefix}_reasoning"]        = None
    out["{prefix}_score"]            = None
    out["{prefix}_rating"]           = None
    out["{prefix}_dominant_signals"] = None

    failed_chunks = []  # list of (chunk_idx, start_pos, end_pos)

    with tqdm(total=n_chunks, unit="chunk", colour="cyan") as pbar:
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk = out.iloc[start:end]

            result = label_chunk(chunk)

            if result is not None:
                for _, row in result.iterrows():
                    pos = int(row["row_id"])
                    if 0 <= pos < total:
                        out.at[pos, "{prefix}_reasoning"]        = row["reasoning"]
                        out.at[pos, "{prefix}_score"]            = int(row["score"]) if pd.notna(row["score"]) else None
                        out.at[pos, "{prefix}_rating"]           = row["rating"]
                        out.at[pos, "{prefix}_dominant_signals"] = row["dominant_signals"]
            else:
                failed_chunks.append((i, start, end))

            # Save progress every 10 chunks
            if (i + 1) % 10 == 0:
                out.to_csv(OUTPUT_CSV, index=False)
                labeled_so_far = out["{prefix}_score"].notna().sum()
                print(f"\\n  Checkpoint saved after chunk {{i+1}}/{{n_chunks}} — {{labeled_so_far:,}}/{{total:,}} labeled")

            pbar.update(1)

    labeled = out["{prefix}_score"].notna().sum()
    print(f"\\nPass 1 done — labeled: {{labeled:,}}  |  failed chunks: {{len(failed_chunks)}}")

    # Retry failed chunks
    if failed_chunks:
        print(f"\\nRetrying {{len(failed_chunks)}} failed chunks …")
        still_failed = []

        with tqdm(total=len(failed_chunks), unit="chunk", colour="yellow") as pbar:
            for chunk_idx, start, end in failed_chunks:
                chunk = out.iloc[start:end]
                result = label_chunk(chunk)

                if result is not None:
                    for _, row in result.iterrows():
                        pos = int(row["row_id"])
                        if 0 <= pos < total:
                            out.at[pos, "{prefix}_reasoning"]        = row["reasoning"]
                            out.at[pos, "{prefix}_score"]            = int(row["score"]) if pd.notna(row["score"]) else None
                            out.at[pos, "{prefix}_rating"]           = row["rating"]
                            out.at[pos, "{prefix}_dominant_signals"] = row["dominant_signals"]
                else:
                    still_failed.append((chunk_idx, start, end))
                pbar.update(1)

        print(f"Retry done — recovered: {{len(failed_chunks) - len(still_failed):,}}  |  still missing: {{len(still_failed):,}}")

    out.to_csv(OUTPUT_CSV, index=False)
    labeled = out["{prefix}_score"].notna().sum()
    print(f"\\nSaved → {{OUTPUT_CSV}}  ({{labeled:,}}/{{total:,}} labeled)")
    return out
'''
    return code


def generate_mistral():
    cells = [
        markdown_cell("# Imports", metadata={"id": "tukWH86x_4cc"}),
        code_cell(
            "import io\n"
            "import threading, time, json, re\n"
            "import pandas as pd\n"
            "from collections import deque\n"
            "from queue import Queue, Empty\n"
            "from openai import OpenAI\n"
            "from tqdm.auto import tqdm",
            metadata={"id": "3oFF0xiP_Asx"}
        ),
        markdown_cell("# Configuration", metadata={"id": "SnrszFst_7Sm"}),
        code_cell(
            'API_KEY    = "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"\n'
            'MODEL      = "mistralai/mistral-small-4-119b-2603"\n'
            'BASE_URL   = "https://integrate.api.nvidia.com/v1"\n'
            'INPUT_CSV  = "https://github.com/vmahawar/data-science-datasets-collection/raw/main/mental_health.csv"',
            metadata={"id": "ibIpZe9h_Fq5"}
        ),
        code_cell(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            'OUTPUT_CSV = "/content/drive/MyDrive/Mistral_output.csv"',
            metadata={"colab": {"base_uri": "https://localhost:8080/"}, "id": "MDNKQUOKhpy3", "outputId": "84f7cd05-04e6-48ff-9951-45073c69ef9d"}
        ),
        code_cell(
            "RPM_LIMIT   = 28\n"
            "WINDOW_SEC  = 60\n"
            "REST_SEC    = 35\n"
            "CHUNK_SIZE  = 20   # rows per API call (tune based on average text length)",
            metadata={"id": "6W90YVZbDucd"}
        ),
        markdown_cell("# Batch Prompt + LLM For Labeling", metadata={"id": "xErJFpaL__f2"}),
        code_cell(build_core_code("mistral"), metadata={"id": "pjegrEDR_OiJ"}),
        code_cell(
            'if __name__ == "__main__":\n'
            "    df = pd.read_csv(INPUT_CSV)\n"
            '    df_labeled = label_dataframe(df, text_col="text", label_col="label")\n'
            "    print(df_labeled.head())",
            metadata={"id": "c_p91X87_hKX"}
        ),
        code_cell("", metadata={"id": "3FxCUOedESs7"}),
    ]

    nb = build_notebook(cells)
    nb["metadata"]["colab"] = {"provenance": []}

    with open("Phase1_Prompt Engineering/Mistral_Prompt_Labeling.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Generated Mistral notebook")


def generate_qwen3():
    cells = [
        markdown_cell("# Imports", metadata={"id": "tukWH86x_4cc"}),
        code_cell(
            "import io\n"
            "import threading, time, json, re\n"
            "import pandas as pd\n"
            "from collections import deque\n"
            "from queue import Queue, Empty\n"
            "from openai import OpenAI\n"
            "from tqdm.auto import tqdm",
            metadata={"id": "3oFF0xiP_Asx"}
        ),
        markdown_cell("# Configuration", metadata={"id": "SnrszFst_7Sm"}),
        code_cell(
            'API_KEY    = "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"\n'
            'MODEL      = "qwen/qwen3-next-80b-a3b-thinking"\n'
            'BASE_URL   = "https://integrate.api.nvidia.com/v1"\n'
            'INPUT_CSV  = "/home/mabdrabou/Desktop/NLP Project/mental_health.csv"\n'
            'OUTPUT_CSV = "/home/mabdrabou/Desktop/NLP Project/qwen_labels_output.csv"',
            metadata={"id": "ibIpZe9h_Fq5"}
        ),
        code_cell(
            "RPM_LIMIT   = 28\n"
            "WINDOW_SEC  = 60\n"
            "REST_SEC    = 35\n"
            "CHUNK_SIZE  = 20   # rows per API call (tune based on average text length)",
            metadata={"id": "6W90YVZbDucd"}
        ),
        markdown_cell("# Batch Prompt + LLM For Labeling", metadata={"id": "xErJFpaL__f2"}),
        code_cell(build_core_code("qwen"), metadata={"id": "pjegrEDR_OiJ"}),
        code_cell(
            'if __name__ == "__main__":\n'
            "    df = pd.read_csv(INPUT_CSV)\n"
            '    df_labeled = label_dataframe(df, text_col="text", label_col="label")\n'
            "    print(df_labeled.head())",
            metadata={"id": "c_p91X87_hKX"}
        ),
        code_cell("", metadata={"id": "3FxCUOedESs7"}),
    ]

    nb = build_notebook(cells)
    with open("Phase1_Prompt Engineering/Qwen3_Prompt_Labeling.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Generated Qwen3 notebook")


if __name__ == "__main__":
    generate_mistral()
    generate_qwen3()
    print("Done!")
