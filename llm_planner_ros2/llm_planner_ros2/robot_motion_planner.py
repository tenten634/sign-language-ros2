"""Text command → validated motion plan using a local Ollama model.

This file is a self-contained copy of the original llm_planner/robot_motion_planner.py
so that the llm_planner_ros2 package can run standalone on the robot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
from ollama import chat as ollama_chat


OLLAMA_MODEL = "gemma3n:e4b-it-q4_K_M"

CAPABILITIES_PATH = Path(__file__).resolve().parent / "robot_capabilities.json"
LOCATIONS_PATH = Path(__file__).resolve().parent / "locations.json"


def _load_capabilities() -> Dict[str, Any]:
    with CAPABILITIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_locations() -> Dict[str, Any]:
    try:
        with LOCATIONS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"locations": []}


ROBOT_CAPABILITIES: Dict[str, Any] = _load_capabilities()
LOCATIONS: Dict[str, Any] = _load_locations()


@dataclass
class PlanResult:
    raw_text: str
    normalized_text: str
    plan: List[Dict[str, Any]]
    validation_messages: List[str]
    llm_raw_output: str
    error: Optional[str] = None
    clarification: Optional[Dict[str, Any]] = None
    total_time_s: float = 0.0
    normalization_time_s: float = 0.0
    planning_time_s: float = 0.0
    validation_time_s: float = 0.0


class OllamaError(RuntimeError):
    pass


def call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    try:
        response = ollama_chat(model=model, messages=messages, stream=False, options=options)
    except Exception as e:
        raise OllamaError(f"Error calling Ollama chat: {e}") from e
    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str):
        raise OllamaError(f"Unexpected Ollama response format: {response!r}")
    return content


def _strip_json_like_string(text: str) -> str:
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "corrected_text" in obj and isinstance(obj["corrected_text"], str):
            return obj["corrected_text"].strip()
    except Exception:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def normalize_command_text(raw_text: str, *, model: str) -> str:
    if not raw_text.strip():
        return raw_text
    system_prompt = (
        "You are an English command normalization tool.\n"
        "The input is a robot command sentence produced by sign-language recognition, "
        "so it may contain spelling errors, missing or extra spaces, or minor grammar issues.\n"
        "Your job is to rewrite it as the most natural and clear English command "
        "while preserving the original intent EXACTLY.\n"
        "You MUST normalize unit expressions and spacing, for example:\n"
        "- \"1m\", \"1 m\", \"1meter\", \"1 me ter\" -> \"1 meter\"\n"
        "- \"90deg\", \"90 deg\", \"90degre\" -> \"90 degrees\"\n"
        "- \"10sec\", \"10 s\", \"10second\" -> \"10 seconds\"\n"
        "CRITICAL: Do NOT add new actions or parameters that are not in the original command.\n"
        "CRITICAL: Do NOT invent numeric values that are not clearly present in the input.\n"
        "Only correct spelling, spacing, grammar, and word order. Keep all actions and parameters exactly as specified.\n"
        "Return only the corrected command sentence on a single line.\n"
        "Do not add any explanations or comments."
    )
    user_prompt = (
        "Please correct the following command sentence.\n\n"
        f"Original:\n{raw_text}\n\n"
        "Return only the corrected command sentence in English as a single line.\n"
        "Remember: Do NOT add any new actions or parameters."
    )
    reply = call_ollama_chat(system_prompt, user_prompt, model=model, temperature=0.1)
    return _strip_json_like_string(reply)


import re


def _heuristic_fix_asr_command(text: str) -> str:
    """Deterministically fix common ASR errors before LLM normalization (e.g., for ward -> forward)."""
    s = (text or "").strip()
    if not s:
        return s
    s = re.sub(r"\bfor\s+ward\b", "forward", s, flags=re.IGNORECASE)
    return s


ACTION_LIST: List[Tuple[int, str, str]] = []


def _build_action_list() -> List[Tuple[int, str, str]]:
    actions = ROBOT_CAPABILITIES.get("actions", {})
    preferred_order = [
        "move_forward",
        "move_backward",
        "rotate_left",
        "rotate_right",
        "wait",
        "navigate_to",
    ]
    ordered_names = [name for name in preferred_order if name in actions] + [
        name for name in actions.keys() if name not in preferred_order
    ]
    action_list = []
    for idx, action_name in enumerate(ordered_names, start=1):
        action_spec = actions.get(action_name, {})
        desc = action_spec.get("description", action_name)
        action_list.append((idx, action_name, desc))
    return action_list


def format_capabilities_for_prompt() -> str:
    """Generate compact capabilities description string for LLM prompts."""
    global ACTION_LIST
    ACTION_LIST = _build_action_list()
    actions = ROBOT_CAPABILITIES.get("actions", {})

    lines: List[str] = []
    lines.append("AVAILABLE ACTIONS (select by number):")
    lines.append("")
    for idx, action_name, desc in ACTION_LIST:
        action = actions.get(action_name, {})
        params = action.get("params") or {}
        lines.append(f"{idx}. {desc}")
        if params:
            param_parts = []
            for pname, spec in params.items():
                ptype = spec.get("type", "float")
                mn = spec.get("min")
                mx = spec.get("max")
                range_str = ""
                if mn is not None or mx is not None:
                    if mn is not None and mx is not None:
                        range_str = f" ({mn}-{mx})"
                    elif mn is not None:
                        range_str = f" (min {mn})"
                    else:
                        range_str = f" (max {mx})"
                param_parts.append(f"{pname}: {ptype}{range_str}")
            lines.append(f"   Parameters: {', '.join(param_parts)}")
        lines.append("")

    lines.append("IMPORTANT RULES:")
    lines.append("- Use 'action_id' field with the number from the list above.")
    lines.append("- Only include parameters that are explicitly present in the command.")
    lines.append("- If a required parameter is missing, do not guess; request clarification instead.")
    return "\n".join(lines)


def _format_locations_for_prompt() -> str:
    """Generate human-readable location list and intent hints for the LLM."""
    locations = LOCATIONS.get("locations", [])
    if not locations:
        return ""

    lines: List[str] = []
    lines.append("AVAILABLE LOCATIONS:")
    for loc in locations:
        loc_id = loc.get("id", "")
        name = loc.get("name", loc_id)
        aliases = loc.get("aliases", [])
        desc = loc.get("description", "")
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        desc_str = f" - {desc}" if desc else ""
        lines.append(f"- {loc_id}: {name}{alias_str}{desc_str}")

    lines.append("")
    lines.append(
        "When the user expresses an intention instead of a direct location (for example "
        "\"I want to eat\" or \"I want to sleep\"), you may map the intention to the "
        "most appropriate location ID from the list above (for example kitchen/dining_room "
        "for eating, bedroom for sleeping), but ONLY if such a location clearly exists."
    )
    lines.append(
        "Do NOT invent new location IDs. If no suitable location exists in the list, "
        "omit the navigate_to action instead of guessing."
    )
    return "\n".join(lines)


def _map_action_id_to_name(action_id: Any) -> Optional[str]:
    global ACTION_LIST
    if not ACTION_LIST:
        ACTION_LIST = _build_action_list()
    try:
        idx = int(action_id)
        for list_idx, action_name, _ in ACTION_LIST:
            if list_idx == idx:
                return action_name
    except (ValueError, TypeError):
        pass
    return None


_PARAM_CONFIGS = {
    "distance_m": {
        "keywords": [
            "meter",
            "meters",
            "metre",
            "metres",
            "m",
            "mtr",
        ],
        "unit": "meters",
    },
    "angle_deg": {
        "keywords": [
            "degree",
            "degrees",
            "deg",
            "degre",
            "°",
        ],
        "unit": "degrees",
    },
    "duration_s": {
        "keywords": [
            "second",
            "seconds",
            "sec",
            "secs",
            "s",
        ],
        "unit": "seconds",
    },
}

_NUMERIC_PARAMS = set(_PARAM_CONFIGS.keys())

_PARAM_DESCRIPTIONS: Dict[str, str] = {
    "distance_m": "distance in meters",
    "angle_deg": "rotation angle in degrees",
    "duration_s": "duration in seconds",
}

_WORD_TO_NUMBER = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "ninety": 90.0,
    "one hundred eighty": 180.0,
    "one hundred and eighty": 180.0,
    "forty-five": 45.0,
    "forty five": 45.0,
}


def _extract_param_values(command_lower: str, numbers: List[str], keywords: List[str]) -> List[float]:
    values = []
    used_numbers = set()
    for num_str in numbers:
        if num_str in used_numbers:
            continue
        try:
            num_val = float(num_str)
            for keyword in keywords:
                pattern = rf"\b{re.escape(num_str)}\s*{re.escape(keyword)}|{re.escape(keyword)}\s*{re.escape(num_str)}"
                if re.search(pattern, command_lower):
                    values.append(num_val)
                    used_numbers.add(num_str)
                    break
        except ValueError:
            pass
    for word, num_val in _WORD_TO_NUMBER.items():
        if word in command_lower:
            for keyword in keywords:
                pattern = rf"\b{word}\s+{re.escape(keyword)}|{re.escape(keyword)}\s+{word}"
                if re.search(pattern, command_lower):
                    values.append(num_val)
                    break
    return values


def _extract_parameters_from_command(command_text: str) -> Dict[str, List[float]]:
    # Normalize spacing a bit but keep original for patterns that rely on it
    command_lower = command_text.lower()
    numbers = re.findall(r"\b(\d+\.?\d*)\b", command_text)
    extracted = {
        param_name: _extract_param_values(command_lower, numbers, config["keywords"])
        for param_name, config in _PARAM_CONFIGS.items()
    }
    # Also handle compact forms like "1m", "1meter", "1sec"
    compact_patterns = {
        "distance_m": r"(\d+\.?\d*)\s*(?:m|meter|meters|metre|metres|mtr)\b",
        "angle_deg": r"(\d+\.?\d*)\s*(?:deg|degree|degrees|degre|°)\b",
        "duration_s": r"(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds)\b",
    }
    for pname, pattern in compact_patterns.items():
        for match in re.finditer(pattern, command_lower):
            try:
                val = float(match.group(1))
            except ValueError:
                continue
            extracted.setdefault(pname, [])
            if not extracted[pname]:
                extracted[pname].append(val)

    cm_pattern = r"(\d+\.?\d*)\s*(?:centimeter|centimetre|centimeters|centimetres|cm)\b"
    meters_pattern = r"(\d+\.?\d*)\s*(?:meter|metre|meters|metres|m|mtr)\b"
    meters_match = re.search(meters_pattern, command_lower)
    if meters_match:
        meters_val = float(meters_match.group(1))
        after_meters = command_lower[meters_match.end() :]
        cm_match = re.search(cm_pattern, after_meters)
        if cm_match:
            cm_val = float(cm_match.group(1))
            combined_dist = meters_val + (cm_val / 100.0)
            if "distance_m" in extracted and extracted["distance_m"]:
                extracted["distance_m"][0] = combined_dist
            else:
                extracted["distance_m"] = [combined_dist]
    return extracted


def _is_parameter_mentioned_in_command(
    param_name: str, param_value: float, command_text: str, extracted_params: Optional[Dict[str, List[float]]] = None
) -> bool:
    if extracted_params is None:
        extracted_params = _extract_parameters_from_command(command_text)
    if param_name in extracted_params:
        for extracted_val in extracted_params[param_name]:
            if abs(extracted_val - param_value) < 0.01:
                return True
    return False


def _coerce_numeric(value: Any, spec: Dict[str, Any], allow_modification: bool = True) -> Tuple[Optional[float], Optional[str], bool]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, f"value {value!r} is not numeric", False
    typ = spec.get("type", "float")
    mn = spec.get("min")
    mx = spec.get("max")
    if mn is not None and num < float(mn):
        return None, f"value {num} is below minimum {mn}", False
    if mx is not None and num > float(mx):
        return None, f"value {num} exceeds maximum {mx}", False
    if typ == "int":
        num = int(num)
    return num, None, False


def validate_plan(raw_plan: List[Dict[str, Any]], command_text: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    actions_cfg = ROBOT_CAPABILITIES.get("actions", {})
    normalized: List[Dict[str, Any]] = []
    messages: List[str] = []
    if not isinstance(raw_plan, list):
        return [], ["plan is not a list"]
    extracted_params = _extract_parameters_from_command(command_text) if command_text else None
    for idx, step in enumerate(raw_plan):
        if not isinstance(step, dict):
            messages.append(f"step {idx}: not an object, skipped")
            continue
        action_id = step.get("action_id")
        action_name = step.get("action")
        action_description = step.get("action_description")
        if action_id is not None:
            action_name = _map_action_id_to_name(action_id)
            if not action_name:
                messages.append(f"step {idx}: invalid action_id {action_id!r}, skipped")
                continue
        elif action_description:
            desc_lower = str(action_description).lower().strip()
            action_name = desc_lower if desc_lower in ROBOT_CAPABILITIES.get("actions", {}) else None
            if not action_name:
                messages.append(f"step {idx}: could not map action description {action_description!r}, skipped")
                continue
        if not action_name or not isinstance(action_name, str):
            messages.append(f"step {idx}: missing 'action_id' field, skipped")
            continue
        if action_name not in actions_cfg:
            messages.append(f"step {idx}: unknown action {action_name!r}, skipped")
            continue
        action_spec = actions_cfg[action_name]
        param_specs = action_spec.get("params") or {}
        norm_step: Dict[str, Any] = {"action": action_name}
        step_msgs: List[str] = []
        for pname, pspec in param_specs.items():
            if pname not in step:
                step_msgs.append(f"missing required parameter {pname!r} for action {action_name!r}")
                continue
            raw_val = step[pname]
            if command_text and pname in _NUMERIC_PARAMS:
                try:
                    if not _is_parameter_mentioned_in_command(pname, float(raw_val), command_text, extracted_params):
                        step_msgs.append(f"parameter {pname!r} with value {raw_val!r} was inferred, removing it")
                        continue
                except (ValueError, TypeError):
                    pass
            if pspec.get("type", "float") in ("float", "int", "number"):
                coerced, warn, _ = _coerce_numeric(raw_val, pspec)
                if coerced is None:
                    detail = warn or "invalid numeric value"
                    step_msgs.append(f"parameter {pname!r} invalid: {detail} (got {raw_val!r})")
                    continue
                norm_step[pname] = coerced
            elif pname == "location" and action_name == "navigate_to":
                if not isinstance(raw_val, str):
                    step_msgs.append(f"parameter {pname!r} must be a string: {raw_val!r}")
                    continue
                valid_ids = [loc.get("id") for loc in LOCATIONS.get("locations", [])]
                if raw_val not in valid_ids:
                    step_msgs.append(f"parameter {pname!r} '{raw_val}' is not a valid location ID")
                    continue
                norm_step[pname] = raw_val
            else:
                norm_step[pname] = raw_val
        if set(norm_step.keys()) == {"action"}:
            step_msgs.append(f"no valid parameters for action {action_name!r}, dropping step")
            messages.append(f"step {idx}: " + "; ".join(step_msgs))
            continue
        for msg in step_msgs:
            messages.append(f"step {idx}: {msg}")
        normalized.append(norm_step)
    return normalized, messages


def _extract_json_array(text: str) -> str:
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return s
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    if s.startswith("{") and s.endswith("}"):
        return f"[{s}]"
    return s


def _parse_llm_output(llm_raw: str) -> Optional[Dict[str, Any]]:
    s = llm_raw.strip()
    if "```json" in s:
        start = s.find("```json")
        end = s.find("```", start + 7)
        if end != -1:
            s = s[start + 7 : end].strip()
    elif "```" in s:
        start = s.find("```")
        end = s.find("```", start + 3)
        if end != -1:
            s = s[start + 3 : end].strip()
    if "needs_clarification" in s and s.startswith("{"):
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end > start:
                clarification_data = json.loads(s[start : end + 1])
                if isinstance(clarification_data, dict) and clarification_data.get("needs_clarification"):
                    return clarification_data
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        json_str = _extract_json_array(s)
        data = json.loads(json_str)
        if isinstance(data, dict) and data.get("needs_clarification"):
            return data
        if isinstance(data, list) and len(data) == 1:
            item = data[0]
            if isinstance(item, dict) and item.get("needs_clarification"):
                return item
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def plan_from_normalized_command(
    normalized_text: str, original_text: Optional[str], *, model: str
) -> Tuple[List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
    # Use the normalized sentence consistently for plan generation, numeric extraction,
    # and unit-bearing number checks so behavior matches the planner LLM input.
    command_for_extraction = normalized_text
    extracted_params = _extract_parameters_from_command(command_for_extraction)
    spec = format_capabilities_for_prompt()
    locations_spec = _format_locations_for_prompt()
    system_prompt = (
        "You are a motion planner. Your task is ONLY to identify actions and basic\n"
        "parameters from the user's command. Another system will handle execution.\n\n"
        f"{spec}\n\n"
        f"{locations_spec}\n\n"
        "OUTPUT FORMAT (very important):\n"
        "- Output ONE ACTION PER LINE.\n"
        "- Each line MUST have the following exact format:\n"
        "  ACTION: <action_id>; DIST: <distance_m or empty>; ANG: <angle_deg or empty>; DUR: <duration_s or empty>; LOC: <location_id or empty>\n"
        "- Use empty value when a parameter is not specified in the command.\n"
        "- Do NOT output JSON. Do NOT output any explanation. Only lines in this format.\n\n"
        "Examples:\n"
        "User: 'move forward 1 meter, then turn right 90 degrees'\n"
        "ACTION: 1; DIST: 1.0; ANG: ; DUR: ; LOC:\n"
        "ACTION: 4; DIST: ; ANG: 90.0; DUR: ; LOC:\n\n"
        "User: 'go to kitchen'\n"
        "ACTION: 6; DIST: ; ANG: ; DUR: ; LOC: kitchen\n\n"
        "User: 'wait 5 seconds'\n"
        "ACTION: 5; DIST: ; ANG: ; DUR: 5.0; LOC:\n\n"
        "CRITICAL RULES:\n"
        "- Only output actions that are clearly mentioned or strongly implied by the command.\n"
        "- Do NOT invent numeric values that are not present in the text.\n"
        "- Use only location IDs that exist in AVAILABLE LOCATIONS.\n"
        "- If the command is unclear, simply omit that action (do NOT output a special error line).\n"
    )
    user_prompt = f"User command:\n{normalized_text}\n\nNow output the actions, one per line, using the required format."
    llm_raw = call_ollama_chat(system_prompt, user_prompt, model=model, temperature=0.1)

    # Parse line-based action format from LLM output
    steps: List[Dict[str, Any]] = []
    line_pattern = re.compile(
        r"^ACTION:\s*(?P<action_id>\d+)\s*;"
        r"\s*DIST:\s*(?P<dist>[^;]*)\s*;"
        r"\s*ANG:\s*(?P<ang>[^;]*)\s*;"
        r"\s*DUR:\s*(?P<dur>[^;]*)\s*;"
        r"\s*LOC:\s*(?P<loc>.*)\s*$",
        re.IGNORECASE,
    )
    for raw_line in llm_raw.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("user command"):
            continue
        m = line_pattern.match(line)
        if not m:
            continue
        action_id_str = m.group("action_id")
        try:
            action_id = int(action_id_str)
        except ValueError:
            continue
        step: Dict[str, Any] = {"action_id": action_id}
        dist_str = m.group("dist").strip()
        ang_str = m.group("ang").strip()
        dur_str = m.group("dur").strip()
        loc_str = m.group("loc").strip()
        if dist_str:
            try:
                step["distance_m"] = float(dist_str)
            except ValueError:
                pass
        if ang_str:
            try:
                step["angle_deg"] = float(ang_str)
            except ValueError:
                pass
        if dur_str:
            try:
                step["duration_s"] = float(dur_str)
            except ValueError:
                pass
        if loc_str:
            step["location"] = loc_str
        steps.append(step)

    clarification: Optional[Dict[str, Any]] = None
    repetition_count = None
    command_lower = command_for_extraction.lower()
    repetition_match = re.search(r"(\d+)\s+times?\b", command_lower)
    if repetition_match:
        try:
            repetition_count = int(repetition_match.group(1))
        except ValueError:
            pass
    final_steps: List[Dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        action_id = step.get("action_id")
        if action_id is None:
            final_steps.append(step)
            continue
        action_name = _map_action_id_to_name(action_id)
        if not action_name:
            final_steps.append(step)
            continue
        actions_cfg = ROBOT_CAPABILITIES.get("actions", {})
        action_spec = actions_cfg.get(action_name, {})
        param_specs = action_spec.get("params") or {}
        final_step: Dict[str, Any] = {"action_id": action_id}
        missing_params = []
        for pname, pspec in param_specs.items():
            if pname == "location":
                if pname in step:
                    final_step[pname] = step[pname]
                else:
                    missing_params.append(pname)
            elif pname in _NUMERIC_PARAMS:
                if pname in extracted_params and extracted_params[pname]:
                    param_values = extracted_params[pname]
                    if step_idx < len(param_values):
                        final_step[pname] = param_values[step_idx]
                    else:
                        final_step[pname] = param_values[0]
                else:
                    param_keywords = _PARAM_CONFIGS[pname]["keywords"]
                    has_number_near_keyword = False
                    for keyword in param_keywords:
                        pattern = rf"\b(\d+\.?\d*)\s*{re.escape(keyword)}|{re.escape(keyword)}\s*(\d+\.?\d*)"
                        if re.search(pattern, command_lower):
                            has_number_near_keyword = True
                            break
                    if not has_number_near_keyword:
                        missing_params.append(pname)
        if missing_params:
            clarification = _create_clarification(
                action_name,
                action_spec,
                missing_params[0],
                original_text or normalized_text,
            )
            return [], llm_raw, clarification
        if repetition_count and repetition_count > 1 and action_name in ("move_forward", "move_backward"):
            for _ in range(repetition_count):
                final_steps.append(final_step.copy())
        else:
            final_steps.append(final_step)
    return final_steps, llm_raw, None


def _create_clarification(
    action_name: str, action_spec: Dict[str, Any], missing_param: str, original_command: str
) -> Dict[str, Any]:
    action_desc = action_spec.get("description", action_name)
    param_desc = _PARAM_DESCRIPTIONS.get(missing_param, missing_param)
    param_specs = action_spec.get("params") or {}
    param_spec = param_specs.get(missing_param, {})
    return {
        "needs_clarification": True,
        "question": f"Please specify the {param_desc}",
        "type": "missing_parameter",
        "missing_parameter": missing_param,
        "action_name": action_name,
        "action_description": action_desc,
        "original_command": original_command,
        "param_spec": param_spec,
    }


def run_pipeline(raw_text: str, *, model: str) -> PlanResult:
    start_time = time.time()
    normalized = raw_text
    error: Optional[str] = None
    llm_raw = ""
    raw_plan: List[Dict[str, Any]] = []
    validated_plan: List[Dict[str, Any]] = []
    messages: List[str] = []
    clarification: Optional[Dict[str, Any]] = None
    normalization_time = 0.0
    planning_time = 0.0
    validation_time = 0.0
    try:
        norm_start = time.time()
        pre_fixed = _heuristic_fix_asr_command(raw_text)
        normalized = normalize_command_text(pre_fixed, model=model)
        normalization_time = time.time() - norm_start
        plan_start = time.time()
        raw_plan, llm_raw, clarification = plan_from_normalized_command(normalized, original_text=raw_text, model=model)
        planning_time = time.time() - plan_start
        if clarification is None:
            valid_start = time.time()
            validated_plan, messages = validate_plan(raw_plan, command_text=normalized)
            validation_time = time.time() - valid_start
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    total_time = time.time() - start_time
    return PlanResult(
        raw_text=raw_text,
        normalized_text=normalized,
        plan=validated_plan,
        validation_messages=messages,
        llm_raw_output=llm_raw,
        error=error,
        clarification=clarification,
        total_time_s=total_time,
        normalization_time_s=normalization_time,
        planning_time_s=planning_time,
        validation_time_s=validation_time,
    )


def save_plan_to_file(result: PlanResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"plan_{timestamp}.json"
    out_path = output_dir / filename
    payload = {
        "raw_text": result.raw_text,
        "normalized_text": result.normalized_text,
        "plan": result.plan,
        "validation_messages": result.validation_messages,
        "error": result.error,
        "clarification": result.clarification,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan robot motions from natural-language commands using a local Ollama model."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=OLLAMA_MODEL,
        help="Ollama model name to use (default: %(default)s)",
    )
    args = parser.parse_args()
    model_name = args.model
    print("=== Robot Motion Planner (Ollama) ===")
    print("Model :", model_name)
    print("Enter a command (empty line to quit).")
    while True:
        try:
            text = input("\nCommand> ").strip()
        except EOFError:
            break
        if not text:
            break
        result = run_pipeline(text, model=model_name)
        if not result.plan and not result.error:
            print(f"\n[DEBUG] LLM raw output: {result.llm_raw_output[:200]}...")
        while result.clarification:
            clarification = result.clarification
            clarification_type = clarification.get("type", "")
            action_desc = clarification.get("action_description", "")
            original_cmd = clarification.get("original_command", text)
            missing_param = clarification.get("missing_parameter", "")
            question = clarification.get("question", "Please provide more information.")
            param_spec = clarification.get("param_spec", {})
            print(f"\nOriginal command: {original_cmd}")
            if action_desc:
                action_short = action_desc.split(".")[0] if "." in action_desc else action_desc
                print(f"Action: {action_short}")
            print(f"\nQuestion: {question}")
            if param_spec:
                constraint_parts = []
                if "min" in param_spec and "max" in param_spec:
                    constraint_parts.append(f"Range: {param_spec['min']} - {param_spec['max']}")
                elif "min" in param_spec:
                    constraint_parts.append(f"Min: {param_spec['min']}")
                elif "max" in param_spec:
                    constraint_parts.append(f"Max: {param_spec['max']}")
                if "step" in param_spec:
                    constraint_parts.append(f"Step: {param_spec['step']}")
                if constraint_parts:
                    print(f"  Info: {' | '.join(constraint_parts)}")
            if clarification_type == "ambiguous_location" and "options" in clarification:
                options = clarification.get("options", [])
                print("\nAvailable options:")
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
            try:
                response = input("\nYour answer: ").strip()
            except EOFError:
                break
            if not response:
                print("Cancelled.")
                break
            if clarification_type == "missing_parameter":
                if param_spec and missing_param in _NUMERIC_PARAMS:
                    extracted_value = _extract_value_from_response(response, missing_param)
                    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", extracted_value)
                    if not numbers:
                        print(
                            f"\nWarning: Could not extract a numeric value from '{response}'. Please provide a numeric value."
                        )
                        continue
            if clarification_type == "missing_parameter":
                new_command = _build_clarified_command(missing_param, action_desc, original_cmd, response)
            elif clarification_type == "ambiguous_location":
                if response.isdigit():
                    options = clarification.get("options", [])
                    idx = int(response) - 1
                    if 0 <= idx < len(options):
                        selected_location = options[idx]
                        new_command = f"go to {selected_location}"
                    else:
                        print("Invalid selection. Please try again.")
                        continue
                else:
                    new_command = f"go to {response}"
            else:
                new_command = f"{text} {response}"
            print(f"\nProcessing clarified command: {new_command}")
            result = run_pipeline(new_command, model=model_name)
        print("\n--- Pipeline result ---")
        print(f"Raw text       : {result.raw_text!r}")
        print(f"Normalized text: {result.normalized_text!r}")
        print(f"Error          : {result.error!r}")
        print("Plan (validated):")
        print(json.dumps(result.plan, ensure_ascii=False, indent=2))
        print("\n--- Timing ---")
        print(f"Total time     : {result.total_time_s:.2f}s")
        print(f"  Normalization: {result.normalization_time_s:.2f}s")
        print(f"  Planning     : {result.planning_time_s:.2f}s")
        print(f"  Validation   : {result.validation_time_s:.2f}s")
        if result.validation_messages:
            print("\nValidation messages:")
            for msg in result.validation_messages:
                print(f"- {msg}")
        out_dir = Path(__file__).resolve().parent / "plans"
        out_path = save_plan_to_file(result, out_dir)
        print(f"\nPlan saved to: {out_path}")


if __name__ == "__main__":
    main()

