#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# === HW1 SOLUTION START ===
MODEL_NAME = "deepseek-v4-flash-vision-exp"
MAX_CONCURRENCY = 8
DEFAULT_READS = 5
TOLERANCE = Decimal("0.05")
NO_ITEMS = Decimal("999999.99")

# Transcription only. The model is never told what the numbers must add up to:
# given an arithmetic identity to satisfy, it will misread a second line to make
# a first misreading balance, which hides the error instead of exposing it.
EXTRACTION_SYSTEM_PROMPT = """You are transcribing one Hong Kong supermarket receipt.
Copy what is printed on it. Do not calculate anything.

Report these fields as JSON:
- "items": every printed line that ADDS to the bill -- each product line, and also bag charges, deposits and service charges -- as a positive number, one entry per printed line, in order.
- "discounts": every printed line that REDUCES the bill -- discount, promotion, coupon, "% OFF", "MEMBER PRICE", "SAVE", "REDEEM" and similar. Give each one as {{"label": the text printed on that line, "amount": the amount in the right-hand column as a positive number}}, one entry per printed line. A ROUNDING line is not a discount.
  A discount is usually printed on the line directly BELOW the item it applies to, labelled like "Buy 2 Save $6", "MB APP UPGRADE -$10" or "5% OFF". Copy its "label" character for character, and take its "amount" from the right-hand column.
- "subtotal": the SUBTOTAL line exactly as printed.
- "rounding": the ROUNDING / ADJUSTMENT line exactly as printed, keeping its sign. Use 0 when the receipt has no such line.
- "final_paid": the payment line (OCTOPUS, CASH, VISA, MASTER, ALIPAY, AMOUNT DUE, TOTAL ...) exactly as printed.

Rules:
- Transcribe only. Do NOT add, subtract, check or reconcile any total.
- Do NOT adjust any number so that the receipt adds up. If the printed numbers do not appear to agree, report them exactly as printed anyway.
- One entry per printed line. Never merge two lines, never invent a line, never drop a line.
- Numbers only: no currency symbol, no thousands separator, two decimals.
- Reply with a single JSON object and nothing else:
{{"items": [], "discounts": [{{"label": "", "amount": 0.00}}], "subtotal": 0.00, "rounding": 0.00, "final_paid": 0.00}}
"""

BASE_INSTRUCTION = "Transcribe this receipt's lines and totals as JSON, following the field definitions exactly."


def _to_decimal(value: Any) -> Decimal | None:
    """Best-effort conversion of a model-supplied amount to Decimal."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value)).quantize(Decimal("0.01"))
        except InvalidOperation:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace("HK$", "").replace("$", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        amount = Decimal(match.group(0)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _amount_list(raw: Any, rounding: Decimal, drop_rounding: bool) -> list[Decimal]:
    """Normalise a model-supplied list of amounts to positive Decimals."""
    if isinstance(raw, (int, float, str, Decimal)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    amounts: list[Decimal] = []
    for entry in raw:
        if isinstance(entry, dict):
            entry = entry.get("amount", entry.get("value", entry.get("price")))
        amount = _to_decimal(entry)
        if amount is None or amount == 0:
            continue
        amount = abs(amount)
        # A ROUNDING line copied into "discounts" is identifiable because it
        # equals |rounding|; it must not inflate the discount total.
        if drop_rounding and rounding != 0 and amount == abs(rounding):
            continue
        amounts.append(amount)
    return amounts


def _label_confirms(label: Any, amount: Decimal) -> bool:
    """True when the line's own text repeats the amount printed in the column.

    Most Hong Kong supermarket discounts are labelled with their own value --
    "Buy 3 Save $9.8", "MB APP UPGRADE -$10" -- so the label independently
    confirms the column. A read that misreads the column loses that agreement.
    The model is never told this is being checked, so it has no way to satisfy
    the check by writing a label to match a number it invented.
    """
    if not isinstance(label, str):
        return False
    for match in re.finditer(r"\d+(?:\.\d+)?", label):
        try:
            if Decimal(match.group(0)).quantize(Decimal("0.01")) == amount:
                return True
        except InvalidOperation:
            continue
    return False


def _discount_entries(raw: Any, rounding: Decimal) -> tuple[list[Decimal], int]:
    """Return the discount amounts, and how many their labels confirm."""
    if isinstance(raw, (int, float, str, Decimal)):
        raw = [raw]
    if not isinstance(raw, list):
        return [], 0
    amounts: list[Decimal] = []
    confirmed = 0
    for entry in raw:
        label = None
        if isinstance(entry, dict):
            label = entry.get("label", entry.get("text", entry.get("description")))
            entry = entry.get("amount", entry.get("value", entry.get("price")))
        amount = _to_decimal(entry)
        if amount is None or amount == 0:
            continue
        amount = abs(amount)
        # A ROUNDING line copied in here is identifiable: it equals |rounding|.
        if rounding != 0 and amount == abs(rounding):
            continue
        amounts.append(amount)
        if _label_confirms(label, amount):
            confirmed += 1
    return amounts, confirmed


def _parse_record(payload: Any) -> dict[str, Decimal] | None:
    """Turn one transcription into Decimal fields, with its own diagnostics."""
    if isinstance(payload, BaseException) or not isinstance(payload, dict):
        return None

    subtotal = _to_decimal(payload.get("subtotal"))
    rounding = _to_decimal(payload.get("rounding")) or Decimal("0.00")
    final_paid = _to_decimal(payload.get("final_paid"))
    if subtotal is None and final_paid is None:
        return None
    if subtotal is None:
        subtotal = final_paid - rounding
    if final_paid is None:
        final_paid = subtotal + rounding

    discounts, confirmed = _discount_entries(payload.get("discounts"), rounding)
    items = _amount_list(payload.get("items"), rounding, drop_rounding=False)
    discount_total = sum(discounts, Decimal("0.00"))
    items_total = sum(items, Decimal("0.00"))

    return {
        "final_paid": final_paid,
        "subtotal": subtotal,
        "discount_total": discount_total,
        "items_total": items_total,
        "confirmed": confirmed,
        # Used only to break ties between disagreeing reads, never shown to the
        # model: how far this read's item lines are from explaining its subtotal.
        "imbalance": (abs(items_total - discount_total - subtotal) if items
                      else NO_ITEMS),
        "payment_ok": abs(subtotal + rounding - final_paid) <= TOLERANCE,
    }


def _merge(candidates: list[dict[str, Decimal]]) -> dict[str, Decimal] | None:
    """Combine several independent reads of one receipt into one answer.

    A misread digit is not reproducible, so the value that two reads agree on is
    far more likely than the one that appeared once. Fields are voted on
    separately because different reads spoil different fields.
    """
    from collections import Counter

    if not candidates:
        return None
    trusted = [c for c in candidates if c["payment_ok"]] or candidates

    merged: dict[str, Decimal] = {}
    agreement = []
    for field in ("final_paid", "subtotal"):
        value, count = Counter(c[field] for c in trusted).most_common(1)[0]
        if count == 1:
            value = min(trusted, key=lambda c: c["imbalance"])[field]
        merged[field] = value
        agreement.append(count)

    # Query 2 needs the bill before any discount, and the receipt states it twice:
    # as subtotal + discounts, and as the item lines at their printed prices. A
    # read whose two versions agree got the whole receipt right, so those reads
    # are ranked first -- then the label checksum, then the vote. Nothing here is
    # ever shown to the model, so it cannot write numbers to satisfy it.
    consistent = [c for c in trusted if c["imbalance"] <= TOLERANCE]
    pool = consistent or trusted
    best = max(c["confirmed"] for c in pool)
    pool = [c for c in pool if c["confirmed"] == best]

    gross_values = [c["subtotal"] + c["discount_total"] for c in pool]
    if not consistent:
        # Nothing self-consistent: let both estimates vote and take the value
        # the receipt produced most often, rather than trusting one of them.
        gross_values += [c["items_total"] for c in pool if c["items_total"] > 0]
    gross, count = Counter(gross_values).most_common(1)[0]
    agreement.append(count)

    merged["gross"] = gross
    merged["discount_total"] = gross - merged["subtotal"]
    merged["items_total"] = min(pool, key=lambda c: c["imbalance"])["items_total"]
    merged["confirmed"] = best
    merged["consistent"] = len(consistent)
    merged["pool"] = len(pool)
    merged["reads"] = len(candidates)
    merged["agreement"] = min(agreement)
    return merged


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    prompt -> vision model -> JSON. One receipt per call; every total is added
    up in Python, where it cannot be hallucinated.
    """
    ### YOUR CODE HERE
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
        max_retries=3,
        timeout=180,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            (
                "human",
                [
                    {"type": "text", "text": "{instruction}"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string."""
    ### YOUR CODE HERE
    import os

    debug = os.environ.get("HW1_DEBUG") == "1"
    reads = max(1, int(os.environ.get("HW1_READS", DEFAULT_READS)))
    concurrency = max(1, int(os.environ.get("HW1_CONCURRENCY", MAX_CONCURRENCY)))
    candidates: dict[int, list[dict[str, Decimal]]] = {i: [] for i in range(len(images))}
    failures: dict[int, int] = {i: 0 for i in range(len(images))}

    try:
        if chain is not None and images:
            urls = [image_data_url(path) for path in images]
            # Every read of every receipt goes into one batch, so all of them are
            # in flight together instead of one round trip per retry.
            payloads = [
                {"instruction": BASE_INSTRUCTION, "image_url": url}
                for _ in range(reads)
                for url in urls
            ]
            owners = [index for _ in range(reads) for index in range(len(images))]
            raw = chain.batch(
                payloads,
                config={"max_concurrency": concurrency},
                return_exceptions=True,
            )
            for index, item in zip(owners, raw):
                record = _parse_record(item)
                if record is not None:
                    candidates[index].append(record)
                else:
                    failures[index] += 1
    except Exception:  # never crash: results.csv must always be written
        pass

    total_paid = Decimal("0.00")
    total_without_discount = Decimal("0.00")
    for index, path in enumerate(images):
        record = _merge(candidates[index])
        if record is None:
            if debug:
                print(f"[hw1] {path.name}: extraction failed")
            continue
        total_paid += record["final_paid"]
        total_without_discount += record["gross"]
        if debug:
            print(
                f"[hw1] {path.name}: paid={record['final_paid']} "
                f"subtotal={record['subtotal']} discounts={record['discount_total']} "
                f"gross={record['gross']} "
                f"agree={record['agreement']}/{record['reads']} "
                f"lost={failures[index]}/{reads} "
                f"balanced={record['consistent']} labels_ok={record['confirmed']} "
                f"pool={record['pool']} items={record['items_total']}"
            )

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
