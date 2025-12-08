# file: run_llm_filter_gpt5.py
"""
ReFilter LLM filtering step (GPT-5, few-shot functional similarity).

This script:
  1. Loads app descriptions from a JSONL file.
  2. For each target app, reads its top-K candidate apps from a CSV file
     named <target_app_id>.csv in a candidates directory.
  3. Builds a prompt with a structured definition of functional similarity
     and optional few-shot examples.
  4. Calls the OpenAI Responses API (GPT-5) to decide which candidates
     are functionally similar to the target.
  5. Writes results to:
       - outputs.jsonl  (one JSON record per target)

Usage (example):

  python run_llm_filter_gpt5.py \
      --apps_jsonl data/apps_high_downloads.jsonl \
      --candidates_dir data/candidates_linq \
      --output_dir experiments \
      --experiment_name gpt5_2shot_refilter \
      --targets_file config/eval_targets.txt \
      --fewshot_mode 2shot \
      --top_k 30

Requirements:
  - Environment variable OPENAI_API_KEY must be set.
  - apps_jsonl must contain one JSON object per line with at least:
        "App ID" and "Description".
  - candidates_dir must contain, for each target appId, a CSV file:
        <target_app_id>.csv
    with columns:
        appId, rank
"""

import os
import json
import ast
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =====================================================
#           MODEL & GENERATION SETTINGS (FIXED)
# =====================================================

MODEL_NAME = "gpt-5"          # Keep fixed for replication
TEMPERATURE: Optional[float] = None
MAX_OUTPUT_TOKENS = 4096      # Supports 30 candidates with short reasons

# Prompt truncation guards
MAX_TARGET_CHARS = 4000
MAX_CANDIDATE_CHARS = 10000

# =====================================================
#                PROMPT INTRO BLOCK
# =====================================================

INTRO = {
    "input_description": (
        "You receive a single target app (appID + description) and a list of 30 candidate apps "
        "(each with appID + description)."
    ),
    "request": (
        "Decide which candidate apps are functionally similar to the target app and could serve as alternatives. Provide a short reason behind each decision."
    ),
    "definition": (
        "A candidate app is similar to a target app if a user could switch from one to the other without losing the core functionality of the target app.\n"
        
    ),
    "output_description": (
        "Output ONLY a JSON array of objects, one per candidate, exactly in this shape:\n"
        "{\"appid\":\"<candidate_appid>\",\"verdict\":\"similar|not_similar\",\"reason\":\"≤20 words\"}\n"
        "Do not skip any candidate app. No headings, no extra keys, no narration."
    ),
}

# ===================================================
#          FEW-SHOT TOGGLE & EXAMPLES (HARDcoded)
# =====================================================

TWO_SHOT_PAIRS = [
    {
        "target_app_id": "com.feelweel.astro",
        "candidate_app_id": "com.labyrinthos.app",
        "verdict": "similar",
        "reason":"Core functionality is horoscope, tarot, numerology, candidate app is a tarot app with limited astrology and horoscopes, it is similar",
    },
    {
        "target_app_id": "com.dhanrajchavan.unmutify",
        "candidate_app_id": "com.addevice.translate",
        "verdict": "not_similar",
        "reason":"Core functionality is text to speech, candidate is a translating app",
    },
]


FOUR_SHOT_PAIRS = [
    {
        "target_app_id": "com.feelweel.astro",
        "candidate_app_id": "com.labyrinthos.app",
        "verdict": "similar",
        "reason":"Core functionality is astrology, horoscope, tarot, numerology, candidate app is a tarot app with limited astrology and horoscopes, it is similar",
    },
    {
        "target_app_id": "com.massimobiolcati.irealb",
        "candidate_app_id": "com.songsterr",
        "verdict": "similar",
        "reason":"Core functionality is backing tracks for music practice, candidate app contains this feature",
    },
    {
        "target_app_id": "com.snorelab.app",
        "candidate_app_id": "com.samvmisti.noisechecker",
        "verdict": "not_similar",
        "reason":"Core functionality is tracking snoring, candidate app is a sound meter for noise",
    },
    {
        "target_app_id": "com.dhanrajchavan.unmutify",
        "candidate_app_id": "com.addevice.translate",
        "verdict": "not_similar",
        "reason":"Core functionality is text to speech, candidate is a translating app",
    },
]


SIX_SHOT_PAIRS = [
    {
        "target_app_id": "com.feelweel.astro",
        "candidate_app_id": "com.labyrinthos.app",
        "verdict": "similar",
        "reason":"Core functionality is astrology, horoscope, tarot, numerology, candidate app is a tarot app with limited astrology and horoscopes, it is similar",
    },
    {
        "target_app_id": "com.massimobiolcati.irealb",
        "candidate_app_id": "com.songsterr",
        "verdict": "similar",
        "reason":"Core functionality is backing tracks for music practice, candidate app contains this feature",
    },
    {
        "target_app_id": "io.ivoca.conversationspanish",
        "candidate_app_id": "com.QotoQot.ConjuGato",
        "verdict": "similar",
        "reason":"Core functionality is teaching Spanish, candidate app also teaches Spanish",
    },
    {
        "target_app_id": "com.dhanrajchavan.unmutify",
        "candidate_app_id": "com.cliffweitzman.speechify2",
        "verdict": "similar",
        "reason":"Core functionality is text to speech, candidate also offers text to speech",
    },
    {
        "target_app_id": "com.snorelab.app",
        "candidate_app_id": "ipnossoft.rma.free",
        "verdict": "not_similar",
        "reason":"Core functionality is tracking snoring, candidate app's tracker does not track snoring",
    },
    {
        "target_app_id": "com.dhanrajchavan.unmutify",
        "candidate_app_id": "com.addevice.translate",
        "verdict": "not_similar",
        "reason":"Core functionality is text to speech, candidate is a translating app",
    },
]

FEWSHOT_MODES: Dict[str, List[dict]] = {
    "none": [],
    "2shot": TWO_SHOT_PAIRS,
    "4shot": FOUR_SHOT_PAIRS,
    "6shot": SIX_SHOT_PAIRS,
}

# =====================================================
#                    DATA CLASSES
# =====================================================

@dataclass
class IntroMessageParts:
    input_description: str
    request: str
    definition: str
    output_description: str


@dataclass
class ExperimentConfig:
    experiment_name: str
    apps_jsonl: str
    candidates_dir: str
    output_root: str
    targets: List[str]
    fewshot_mode: str
    top_k: int
    # Fixed model settings (still stored for reproducibility)
    model_name: str = MODEL_NAME
    temperature: Optional[float] = TEMPERATURE
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    max_target_chars: int = MAX_TARGET_CHARS
    max_candidate_chars: int = MAX_CANDIDATE_CHARS

# =====================================================
#                     UTILITIES
# =====================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_app_descriptions(jsonl_path: str) -> Dict[str, str]:
    """
    Load app descriptions from a JSONL file.
    Expected keys (at least one of each pair):
      - app id: "App ID" | "appId" | "app_id"
      - description: "Description" | "description"
    Returns a dict: {app_id: description_string}
    """
    app_desc: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            app_id = obj.get("App ID")
            if not app_id:
                continue
            desc = obj.get("Description")
            app_desc[str(app_id)] = " ".join(str(desc).split())
    return app_desc


def load_top_k_candidates_from_csv(
    candidates_dir: str,
    target_id: str,
    k: int,
) -> List[str]:
    """
    For a given target appId, loads its candidate CSV:
      <candidates_dir>/<target_id>.csv
    and returns the top-K candidate appIds sorted by column 'rank',
    excluding the target app itself if present.
    """
    csv_path = os.path.join(candidates_dir, f"{target_id}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing candidate CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "appId" not in df.columns or "rank" not in df.columns:
        raise ValueError(f"CSV missing required columns appId/rank: {csv_path}")
    df = df.sort_values("rank", ascending=True, kind="stable")
    df = df[df["appId"].astype(str) != str(target_id)]
    return df["appId"].astype(str).head(k).tolist()


def truncate_clean(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split()).strip()
    return text if max_chars is None else text[:max_chars]


def resolve_full_desc(app_desc: Dict[str, str], app_id: str) -> str:
    return app_desc.get(app_id, "")


def render_fewshot_pairs_block(pairs: List[dict], app_desc: Dict[str, str]) -> str:
    """
    Render few-shot examples using full descriptions.
    """
    lines = ["[FEW-SHOT DECISION EXAMPLES]"]
    for i, ex in enumerate(pairs, start=1):
        t_id = ex["target_app_id"]
        c_id = ex["candidate_app_id"]
        verdict = ex["verdict"]
        reason = ex["reason"]
        t_desc_full = resolve_full_desc(app_desc, t_id)
        c_desc_full = resolve_full_desc(app_desc, c_id)
        lines += [
            f"Example {i}",
            f"TARGET (appID: {t_id}) DESCRIPTION: {t_desc_full}",
            f"CANDIDATE (appID: {c_id}) DESCRIPTION: {c_desc_full}",
            f"VERDICT: {verdict}",
            f"REASON: {reason}",
            "",  # blank line
        ]
    return "\n".join(lines)


def build_prompt(
    intro: IntroMessageParts,
    fewshot_pairs: List[dict],
    app_desc_for_shots: Dict[str, str],
    target_id: str,
    t_desc: str,
    candidates: List[Tuple[str, str]],
    max_target_chars: int,
    max_cand_chars: int,
) -> str:
    """
    Build the full LLM prompt for one target app.
    """
    intro_block = (
        "[INPUT DESCRIPTION]\n" + intro.input_description + "\n\n"
        "[REQUEST]\n" + intro.request + "\n\n"
        "[DEFINITION]\n" + intro.definition + "\n\n"
        "[OUTPUT DESCRIPTION]\n" + intro.output_description + "\n"
    )
    chunks = [intro_block]

    if fewshot_pairs:
        chunks.append(render_fewshot_pairs_block(fewshot_pairs, app_desc_for_shots))
        chunks.append("")

    chunks.append(
        f"=== TARGET APP === | appID: {target_id} | Description: "
        f"{truncate_clean(t_desc, max_target_chars)}"
    )
    for i, (cid, cdesc) in enumerate(candidates, start=1):
        chunks.append(
            f"=== CANDIDATE {i} === | appID: {cid} | Description: "
            f"{truncate_clean(cdesc, max_cand_chars)}"
        )
    # DEBUG: print and save prompt (commented out)
    #debug_prompt = "\n".join(chunks)
    #print(debug_prompt)
    #with open("DEBUG_prompt_output.txt", "w", encoding="utf-8") as dbg:
    #    dbg.write(debug_prompt)

    return "\n".join(chunks)

# =====================================================
#              OPENAI RESPONSES API CALL
# =====================================================

def call_openai_list_only(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: Optional[float],
    max_output_tokens: int,
) -> str:
    """
    Calls the OpenAI Responses API and returns the raw text output.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "reasoning": {"effort": "minimal"},
        "tool_choice": "none",
    }
    if isinstance(temperature, (int, float)):
        kwargs["temperature"] = float(temperature)

    resp = client.responses.create(**kwargs)

    # Preferred accessor
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: scan resp.output
    output = getattr(resp, "output", None)
    if isinstance(output, list) and output:
        for item in output:
            content = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            if isinstance(content, list):
                buf: List[str] = []
                for p in content:
                    t = getattr(p, "type", None) or (
                        p.get("type") if isinstance(p, dict) else None
                    )
                    s = getattr(p, "text", None) or (
                        p.get("text") if isinstance(p, dict) else None
                    )
                    if isinstance(s, str) and (t in (None, "output_text", "text")):
                        buf.append(s)
                if buf:
                    return "".join(buf).strip()
            elif isinstance(content, str) and content.strip():
                return content.strip()

    return ""

# =====================================================
#                LLM JSON PARSING
# =====================================================

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s[3:]
        if "\n" in s:
            s = s.split("\n", 1)[1]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def parse_llm_judgments(raw: str) -> List[Dict[str, str]]:
    """
    Parse the model's JSON array:

      [{"appid":"...","verdict":"similar|not_similar","reason":"..."}]

    Returns a list of dicts. Robust to code fences and single quotes.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    txt = _strip_code_fence(raw)

    # If the whole string isn't clean JSON, try to isolate the first bracket block
    if not (txt.strip().startswith("[") and txt.strip().endswith("]")):
        lb, rb = txt.find("["), txt.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            txt = txt[lb:rb + 1].strip()

    # Attempt JSON first
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [dict(x) for x in arr if isinstance(x, (dict,))]
    except Exception:
        pass

    # Try Python literal (handles single quotes)
    try:
        arr = ast.literal_eval(txt)
        if isinstance(arr, list):
            return [dict(x) for x in arr if isinstance(x, (dict,))]
    except Exception:
        pass

    return []

# =====================================================
#                   MAIN RUN LOGIC
# =====================================================

def run_experiment(cfg: ExperimentConfig, intro: IntroMessageParts) -> None:
    """
    Run the LLM filtering experiment over all target apps in cfg.targets.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.output_root, cfg.experiment_name, f"run-{timestamp}")
    ensure_dir(run_dir)

    outputs_path = os.path.join(run_dir, "outputs.jsonl")
    params_path = os.path.join(run_dir, "params.json")

    # Save config
    with open(params_path, "w", encoding="utf-8") as pf:
        json.dump(asdict(cfg), pf, ensure_ascii=False, indent=2)

    client = OpenAI()
    app_desc = load_app_descriptions(cfg.apps_jsonl)

    fewshot_pairs = FEWSHOT_MODES.get(cfg.fewshot_mode, [])
    print(f"Running experiment '{cfg.experiment_name}' with few-shot mode: {cfg.fewshot_mode}")
    print(f"Run directory: {run_dir}")

    records: List[Dict[str, Any]] = []

    with open(outputs_path, "w", encoding="utf-8") as outf:
        for target in tqdm(cfg.targets, desc="Targets"):
            t_desc = app_desc.get(target, "")
            cand_ids = load_top_k_candidates_from_csv(cfg.candidates_dir, target, cfg.top_k)
            cand_pairs = [(cid, app_desc.get(cid, "")) for cid in cand_ids]

            prompt = build_prompt(
                intro=intro,
                fewshot_pairs=fewshot_pairs,
                app_desc_for_shots=app_desc,
                target_id=target,
                t_desc=t_desc,
                candidates=cand_pairs,
                max_target_chars=cfg.max_target_chars,
                max_cand_chars=cfg.max_candidate_chars,
            )

            raw = call_openai_list_only(
                client=client,
                model=cfg.model_name,
                prompt=prompt,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
            )

            arr = parse_llm_judgments(raw)
            by_appid: Dict[str, Dict[str, str]] = {}
            for item in arr:
                appid = str(item.get("appid", "")).strip()
                verdict = str(item.get("verdict", "")).strip().lower()
                reason = str(item.get("reason", "")).strip()
                if appid:
                    by_appid[appid] = {"verdict": verdict, "reason": reason}

            selected = [cid for cid in cand_ids if by_appid.get(cid, {}).get("verdict") == "similar"]

            rec = {
                "target_app_id": target,
                "candidate_ids": cand_ids,
                "selected_ids": selected,
                "raw_response": raw,
                "model": cfg.model_name,
                "experiment": cfg.experiment_name,
                "timestamp": timestamp,
                "fewshot_mode": cfg.fewshot_mode,
            }
            records.append(rec)
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")


    print("Done.")
    print(f"outputs.jsonl:         {outputs_path}")
    print(f"params.json:          {params_path}")

# =====================================================
#                        CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-5 functional similarity filtering for ReFilter."
    )
    parser.add_argument(
        "--apps_jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with app metadata (App ID, Description).",
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help="Directory containing per-target candidate CSVs (<appId>.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Root directory to store experiment outputs (default: experiments/).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="gpt5_2shot_refilter",
        help="Name for this experiment (used in output path).",
    )
    parser.add_argument(
        "--targets_file",
        type=str,
        default=None,
        help="Path to a text file with one target appId per line.",
    )
    parser.add_argument(
        "--target_ids",
        type=str,
        default=None,
        help="Comma-separated list of target appIds (overrides --targets_file if provided).",
    )
    parser.add_argument(
        "--fewshot_mode",
        type=str,
        default="2shot",
        choices=["none", "2shot", "4shot", "6shot"],
        help="Few-shot setting: none, 2shot, 4shot, or 6shot (default: 2shot).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top candidates to consider per target (default: 30).",
    )
    parser.add_argument(
        "--max_targets",
        type=int,
        default=None,
        help="Optional limit on number of target apps to process (for quick tests).",
    )
    return parser.parse_args()


def load_targets_from_args(args: argparse.Namespace) -> List[str]:
    # Highest priority: explicit comma-separated list
    if args.target_ids:
        ids = [s.strip() for s in args.target_ids.split(",") if s.strip()]
        if not ids:
            raise ValueError("No valid target_ids found in --target_ids.")
        return ids

    # Otherwise: targets_file
    if args.targets_file:
        if not os.path.isfile(args.targets_file):
            raise FileNotFoundError(f"targets_file not found: {args.targets_file}")
        ids: List[str] = []
        with open(args.targets_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
        if not ids:
            raise ValueError("No valid target appIds found in targets_file.")
        return ids

    raise ValueError("You must provide either --targets_file or --target_ids.")


def main() -> None:
    args = parse_args()
    targets = load_targets_from_args(args)

    if args.max_targets is not None and len(targets) > args.max_targets:
        targets = targets[: args.max_targets]

    intro = IntroMessageParts(**INTRO)
    cfg = ExperimentConfig(
        experiment_name=args.experiment_name,
        apps_jsonl=args.apps_jsonl,
        candidates_dir=args.candidates_dir,
        output_root=args.output_dir,
        targets=targets,
        fewshot_mode=args.fewshot_mode,
        top_k=args.top_k,
    )
    run_experiment(cfg, intro)


if __name__ == "__main__":
    main()
