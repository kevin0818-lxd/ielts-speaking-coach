import os
import re
import json
import logging
import time
from typing import Dict, Optional, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configure logging
logger = logging.getLogger("ielts_llm")


def _strip_thinking_tags(text: str) -> str:
    """Remove Qwen3-Thinking <think>...</think> blocks and any residual reasoning from output."""
    # Closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Unclosed <think> tag (model hit max_new_tokens mid-thought)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    # Variant tags
    cleaned = re.sub(r"<\|think\|>.*?<\|/think\|>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<\|think\|>.*$", "", cleaned, flags=re.DOTALL).strip()
    # Strip leading reasoning preamble before actual answer
    cleaned = re.sub(r"^(?:Thinking|Let me think|I need to|First,? I).*?\n\n", "", cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def _extract_model_answer(raw: str) -> str:
    """
    Aggressively extract only the rewritten Band-9 answer from model output,
    discarding any thinking/reasoning that leaked through tags or was emitted
    without tags by the fine-tuned adapter.
    """
    raw = _strip_thinking_tags(raw)

    # If </think> is present but regex missed due to encoding, split on it
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()

    # Common patterns where the actual answer starts after reasoning
    answer_markers = [
        "\n\nBand 9 Model Answer:\n",
        "\n\nBand 9 Model Answer\n",
        "\n\nModel Answer:\n",
        "\n\nModel Answer\n",
        "\n\nRewritten:\n",
        "\n\nRewritten Answer:\n",
        "\n\nHere is the rewritten",
        "\n\nHere's the rewritten",
        "\n\nHere is the Band 9",
        "\n\nHere's the Band 9",
        "\n\n---\n",
    ]
    for marker in answer_markers:
        if marker in raw:
            raw = raw.split(marker, 1)[-1].strip()
            break

    # If the output still starts with reasoning-like lines (short analytical sentences
    # before the actual rewrite), heuristic: find the first paragraph that looks like
    # natural speech (longer than 80 chars without bullet/number prefix)
    lines = raw.split("\n\n")
    if len(lines) > 1:
        for i, para in enumerate(lines):
            stripped = para.strip()
            if len(stripped) > 80 and not stripped[0].isdigit() and not stripped.startswith(("-", "*", ">")):
                raw = "\n\n".join(lines[i:]).strip()
                break

    return raw.strip()

# Local fine-tuned Qwen3-Thinking model: resolve path robustly
def _resolve_local_model_path() -> str:
    """Resolve path to local fine-tuned model, supporting env override and multiple candidates."""
    override = os.environ.get("LOCAL_MODEL_PATH")
    if override and os.path.exists(override):
        return override
    _dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_dir, "resources", "llm_finetune", "checkpoint-50"),
        os.path.join(os.getcwd(), "backend", "resources", "llm_finetune", "checkpoint-50"),
        os.path.join(os.getcwd(), "resources", "llm_finetune", "checkpoint-50"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # default for existence check

LOCAL_MODEL_PATH = _resolve_local_model_path()
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_ENABLE_REASONING = os.environ.get("LOCAL_ENABLE_REASONING", "0") == "1"

class DeepSeekEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.client = None
        self.local_model = None
        self.tokenizer = None
        self._device = "cpu"  # device for local model inference

        # Prefer local fine-tuned Qwen3-Instruct model (no API key needed)
        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Loading local fine-tuned Qwen3-Instruct from {LOCAL_MODEL_PATH}...")
            max_retries = 3
            retry_delay = 2.0
            for attempt in range(max_retries):
                try:
                    # Device: CUDA > MPS > CPU
                    if torch.cuda.is_available():
                        device = "cuda"
                        dtype = torch.float16
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                        dtype = torch.float16
                    else:
                        device = "cpu"
                        dtype = torch.float32
                    logger.info(f"Using device: {device}")

                    base_model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_NAME,
                        device_map=None,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=(device == "cpu"),
                    )
                    # Use adapter directly on top of base model (no merge), so reasoning behavior
                    # is governed by the fine-tuned adapter weights during generation.
                    self.local_model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
                    self.local_model.to(device)
                    self.local_model.eval()
                    self._device = device
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        BASE_MODEL_NAME,
                        trust_remote_code=True,
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(
                        f"Local Qwen3-Instruct model loaded successfully (adapter-direct, reasoning={LOCAL_ENABLE_REASONING})."
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"Load attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.exception(
                            f"Failed to load local model from {LOCAL_MODEL_PATH} after {max_retries} attempts"
                        )
        
        if not self.local_model and self.api_key and OpenAI:
            # Fallback to DeepSeek API
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=30.0,
                max_retries=1
            )
        elif not self.local_model:
            if not OpenAI:
                logger.warning("OpenAI library not found. LLM evaluation disabled.")
            elif not self.api_key:
                logger.warning("DEEPSEEK_API_KEY not set. LLM evaluation disabled.")

    def is_available(self) -> bool:
        return self.local_model is not None or self.client is not None

    def evaluate(self, transcript: str, part: str = "Part 1", question: str = "") -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Evaluates the IELTS Speaking transcript using Local Model or DeepSeek-V3.
        Returns: (result_dict, confidence_score)
        """
        if not self.is_available():
            return None, 0.0

        if not transcript or len(transcript.strip()) < 5:
            return None, 0.0

        # Fix for 'NoneType object has no attribute chat' - Ensure client exists for fallback
        # If local model failed initialization but path exists, client might be None
        if not self.client and self.api_key and OpenAI and not self.local_model:
             self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=30.0,
                max_retries=1
            )

        system_prompt = """You are an expert IELTS Speaking Examiner. Your task is to evaluate a candidate's response based on the official public band descriptors.

You must score the response on 4 criteria (0-9 scale, 0.5 increments):
1. Fluency and Coherence (FC)
2. Lexical Resource (LR)
3. Grammatical Range and Accuracy (GRA)
4. Pronunciation (PR)

OFFICIAL BAND DESCRIPTORS (Summary):

BAND 9:
- FC: Fluent with only very occasional repetition or self-correction. Any hesitation is for content, not to find words/grammar.
- LR: Total flexibility and precise use in all contexts. Sustained use of accurate and idiomatic language.
- GRA: Structures are precise and accurate at all times. Native-like 'mistakes' allowed.
- PR: Full range of phonological features. Effortlessly understood.

BAND 8:
- FC: Fluent with only very occasional repetition or self-correction. Hesitation may occasionally be to find words/grammar.
- LR: Wide resource, readily and flexibly used. Skillful use of less common/idiomatic items.
- GRA: Wide range of structures. Majority of sentences are error-free. Occasional non-systematic errors.
- PR: Wide range of features. Sustain appropriate rhythm. Easily understood.

BAND 7:
- FC: Able to keep going and readily produce long turns without noticeable effort. Some hesitation/repetition/self-correction.
- LR: Resource flexibly used. Some ability to use less common/idiomatic items. Effective paraphrase.
- GRA: Range of structures flexibly used. Error-free sentences are frequent.
- PR: Uses range of features but control is variable. Generally understood without much effort.

BAND 6:
- FC: Willing to produce long turns. Coherence may be lost at times due to hesitation/repetition.
- LR: Resource sufficient to discuss topics at length. Meaning is clear despite inappropriacies.
- GRA: Mix of short and complex forms. Errors frequently occur in complex structures.
- PR: Range of features but variable control. Occasional lack of clarity.

BAND 5:
- FC: Relies on repetition/self-correction. Slow speech.
- LR: Resource sufficient for familiar topics but limited flexibility.
- GRA: Basic sentence forms fairly well controlled. Complex structures limited and contain errors.
- PR: Complex speech causes disfluency.

Output strictly in JSON format with this schema:
{
  "scores": {
    "fluency": float,
    "lexical": float,
    "grammar": float,
    "pronunciation": float
  },
  "rationale": {
    "fluency": "string (Reason: ...)",
    "lexical": "string (Reason: ...)",
    "grammar": "string (Reason: ...)",
    "pronunciation": "string (Reason: ...)"
  },
  "overall_rationale": "string (concise summary)",
  "confidence": float (0.0 to 1.0)
}

SCORING RULES:
- Part 1 answers are short (2-4 sentences). Do NOT penalize for brevity in Part 1.
- Part 2 is a long turn (1-2 mins). Look for structure.
- Part 3 is abstract discussion. Look for development of ideas.
- CRITERION-LEVEL FEEDBACK (CHAI STYLE):
  - For each criterion, provide specific EVIDENCE from the transcript.
  - Example format: "Fluency (Band 6.5): Reason: The response sustains multiple clauses with logical sequencing... There are two pauses longer than 800ms..."
  - Be specific about what was done well and what needs improvement.
"""

        user_content = f"""
Context: IELTS Speaking {part}
Question: {question if question else "General Topic"}

Transcript:
"{transcript}"

Evaluate this response.
"""

        # 1. Local Model Inference
        if self.local_model:
            try:
                device = self._device
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=LOCAL_ENABLE_REASONING
                    )
                except TypeError:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                model_inputs = self.tokenizer(
                    [text], return_tensors="pt", padding=True, return_attention_mask=True
                ).to(device)
                attn = model_inputs.attention_mask
                if attn is None:
                    attn = torch.ones_like(model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device)

                with torch.inference_mode():
                    generated_ids = self.local_model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=attn,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.0,
                    )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Extract JSON from potential wrapper text
                try:
                    # Simple heuristic to find JSON
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Local model JSON parsing failed. Falling back to dummy.")
                    return None, 0.0

                confidence = result.get("confidence", 0.6) # Local model default confidence
                return result, confidence

            except Exception as e:
                logger.error(f"Local Model Evaluation failed: {e}")
                # Fallthrough to API if available
                if not self.client and self.api_key and OpenAI:
                     self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.deepseek.com",
                        timeout=30.0,
                        max_retries=1
                    )
                
                if not self.client:
                    return None, 0.0

        # 2. DeepSeek API Inference (Fallback)
        try:
            if not self.client: # Check again
                 if self.api_key and OpenAI:
                     self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.deepseek.com",
                        timeout=30.0,
                        max_retries=1
                    )
                 else:
                    return None, 0.0
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # or deepseek-reasoner
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1, # Low temperature for consistency
                max_tokens=1000,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Retry once if JSON is malformed (e.g. truncated)
                logger.warning("LLM JSON parsing failed. Retrying...")
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=1500, # Increased tokens for retry
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content
                result = json.loads(content)
            
            # Extract confidence (default to 0.5 if not provided)
            confidence = result.get("confidence", 0.5)
            
            # Sanity check scores
            scores = result.get("scores", {})
            for k in ["fluency", "lexical", "grammar", "pronunciation"]:
                if k not in scores:
                    scores[k] = 6.0 # Fallback
                else:
                    # Clamp between 1.0 and 9.0
                    scores[k] = max(1.0, min(9.0, float(scores[k])))
            
            return result, confidence

        except Exception as e:
            logger.error(f"LLM Evaluation failed: {e}")
            return None, 0.0

    def generate_model_answer(
        self,
        transcript: str,
        part: str = "Part 1",
        user_band: float = 6.0,
        trajectory_context: dict = None,
    ) -> str:
        """
        Generates a model answer one band above the candidate's current level.
        Applies the i+1 principle: target_band = min(9.0, user_band + 1.0).
        """
        if not self.is_available():
            return "Model answer generation unavailable (LLM not configured)."

        if not transcript or len(transcript.strip()) < 5:
            return "Transcript too short to generate a model answer."

        target_band = min(9.0, user_band + 1.0)
        target_desc = f"Band {target_band:.1f}"
        trajectory_context = trajectory_context or {}
        trajectory_note = ""
        if trajectory_context:
            step = trajectory_context.get("trajectory_step", {})
            targets = trajectory_context.get("target_concepts", [])
            register_goal = trajectory_context.get("register_goal", "neutral")
            if isinstance(step, dict) or isinstance(targets, list):
                trajectory_note = (
                    f"\nTRAJECTORY CONTEXT:\n"
                    f"- current_step: {json.dumps(step, ensure_ascii=False)}\n"
                    f"- target_concepts: {json.dumps(targets, ensure_ascii=False)}\n"
                    f"- register_goal: {register_goal}\n"
                    f"Prefer naturally using these target concepts when appropriate.\n"
                )

        system_prompt = f"""You are an expert IELTS Speaking Examiner and Tutor.
The candidate currently scores around Band {user_band:.1f}. Your task is to rewrite their response into a {target_desc} Model Answer that sounds like natural, fluent SPOKEN English — not written prose.
Aim precisely at {target_desc} quality — do not overshoot to Band 9 if the target is lower.

GUIDELINES:
1. Preserve the original core meaning and opinion. Do not invent completely new facts unless the original was too vague.
2. Use SPOKEN register vocabulary: favour idiomatic collocations and phrasal verbs (e.g. "come across", "figure out") over formal/academic words. Avoid overly literary or essay-style diction.
3. Use spoken grammar norms: contractions (I'm, don't, it's, I'd), shorter clause chains, occasional fronting ("The food there, it was incredible"), tag questions, and ellipsis. Avoid heavily embedded subordination typical of writing.
4. Include natural discourse markers and stance expressions throughout: openers (Well, Actually, To be honest), connectors (so, I mean, you know, basically), and hedges (I'd say, I suppose, sort of, kind of). Use conversational connectors (so, but, plus) rather than formal conjunctives (therefore, however, moreover).
5. Add occasional spoken discourse bundles for coherence: "the thing is", "what I mean is", "come to think of it", "speaking of which", "first of all".
6. The length should be appropriate for the part (Part 1: 2-4 sentences, Part 2: ~2 mins of speech, Part 3: In-depth discussion).
7. Output ONLY the rewritten spoken text. Do not add explanations, labels, or "Here is the answer:".
8. If trajectory context is provided, incorporate one or two target concepts naturally while preserving meaning.
"""
        system_prompt += trajectory_note

        user_content = f"""
Candidate's Response ({part}):
"{transcript}"

Rewrite this as a {target_desc} Model Answer.
"""

        try:
            if self.local_model is not None and self.tokenizer is not None:
                device = self._device
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                # Disable thinking mode: Qwen3-Thinking will skip <think> and
                # output the answer directly, dramatically reducing token count
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                except TypeError:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                model_inputs = self.tokenizer(
                    [prompt_text], return_tensors="pt", padding=True, return_attention_mask=True
                ).to(device)
                attn = model_inputs.attention_mask
                if attn is None:
                    attn = torch.ones_like(model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device)

                max_tokens = 2048 if "Part 2" in part else 1536
                logger.info(f"Generating model answer with local Qwen model (max_new_tokens={max_tokens})...")
                with torch.inference_mode():
                    generated_ids = self.local_model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                content = _extract_model_answer(content)
                logger.info(f"Model answer generated ({len(content)} chars)")
                if content:
                    return content
                # Retry with thinking enabled: model may output answer after <think> block
                try:
                    logger.info("Local model returned empty, retrying with thinking enabled...")
                    try:
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True
                        )
                    except TypeError:
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    model_inputs = self.tokenizer(
                        [prompt_text], return_tensors="pt", padding=True, return_attention_mask=True
                    ).to(device)
                    attn = model_inputs.attention_mask
                    if attn is None:
                        attn = torch.ones_like(model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device)
                    with torch.inference_mode():
                        generated_ids = self.local_model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=attn,
                            max_new_tokens=2048,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    content = _extract_model_answer(
                        self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    )
                    if content:
                        return content
                except Exception as retry_err:
                    logger.warning(f"Model answer retry failed: {retry_err}")

            if self.client is None and self.api_key and OpenAI:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com",
                    timeout=90.0,
                    max_retries=2
                )

            if self.client is None:
                return "Model answer generation unavailable (LLM not configured)."

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return _extract_model_answer(response.choices[0].message.content.strip())

        except Exception as e:
            logger.error(f"Model Answer generation failed: {e}")
            return "Failed to generate model answer."

    def generate_contextual_vocabulary(
        self,
        transcript: str,
        part: str = "Part 1",
        user_band: float = 6.0,
        trajectory_context: dict = None,
    ) -> list:
        """
        Generates context-aware vocabulary recommendations using LLM.
        Adapts difficulty based on user_band (i+1 principle).
        Returns: List of dicts {original, alternatives, reason}
        """
        if not self.is_available():
            return []

        if not transcript or len(transcript.strip()) < 5:
            return []
            
        target_band = min(9.0, user_band + 1.0)
        target_desc = "Band 9 (Expert)" if target_band >= 8.5 else f"Band {target_band}"
        trajectory_context = trajectory_context or {}
        trajectory_line = ""
        targets = trajectory_context.get("target_concepts", [])
        if isinstance(targets, list) and targets:
            trajectory_line = f"\nTRAJECTORY TARGET CONCEPTS: {json.dumps(targets, ensure_ascii=False)}\nPrefer alternatives aligned with these concepts where natural.\n"

        part_register = {
            "Part 1": "everyday spoken collocations and idiomatic phrases suitable for casual conversation",
            "Part 2": "vivid, descriptive narrative language with sensory collocations and storytelling phrases",
            "Part 3": "abstract discussion vocabulary appropriate for spoken debate (not essay-style academic words)",
        }.get(part, "natural spoken collocations")

        system_prompt = f"""You are an expert IELTS Speaking Tutor specializing in Lexical Resource.
Your task is to identify 3-5 words or phrases in the candidate's response that are below {target_desc} level, and suggest context-aware alternatives appropriate for {target_desc}.

STEP 1 — Identify the topic/domain of the response (e.g. food, travel, technology, relationships, health, education, work, environment).
STEP 2 — For each weak word, suggest alternatives that are natural collocations WITHIN that domain and that a fluent speaker would actually use when talking about this topic.

This is {part} of the IELTS Speaking test. Prefer {part_register}.

OUTPUT FORMAT:
Return a JSON array of objects. Each object must have:
- "original": The exact word/phrase from the text.
- "alternatives": An array of 2-3 better alternatives that are natural collocations for the identified topic.
- "reason": Explain why each alternative fits THIS specific topic context better.

EXAMPLES showing how context changes the recommendation:

Text (about food): "The food there was really good."
Output: [
  {{
    "original": "really good",
    "alternatives": ["absolutely delicious", "incredibly flavourful", "out of this world"],
    "reason": "When describing food, sensory collocations like 'flavourful' or emphatic spoken phrases like 'out of this world' are more natural than generic 'good'."
  }}
]

Text (about a person): "My teacher was really good."
Output: [
  {{
    "original": "really good",
    "alternatives": ["incredibly supportive", "genuinely inspiring", "really dedicated"],
    "reason": "When describing a person, character-trait collocations like 'supportive' or 'dedicated' are more specific and natural than generic 'good'."
  }}
]

GUIDELINES:
1. Do NOT suggest generic thesaurus synonyms that ignore context. Every alternative must be a natural collocation for the specific topic being discussed.
2. Prefer spoken-register alternatives: idiomatic phrases, phrasal verbs, and collocations used in conversation. Avoid overly literary or essay-style words unless the candidate is in Part 3 discussing abstract ideas.
3. Ensure alternatives are grammatically correct when substituted into the original sentence.
4. Suggest words that are slightly challenging but attainable (i+1 level) for a current Band {user_band} speaker.
5. Consider the surrounding sentence structure — if the candidate uses a verb + noun pattern, suggest a better verb + noun collocation, not just a lone adjective.
6. QUALITY OVER QUANTITY: Suggest only 3-5 improvements where a genuinely more natural option exists. Do NOT mechanically upgrade every word. If the candidate's word is already appropriate for the context, skip it.
7. SOUND NATURAL WHEN SPOKEN: Every alternative must sound natural when said aloud in a conversation. Avoid words that would sound stiff, essay-like, or overly formal when spoken.
8. AVOID THESE MECHANICAL MISTAKES: Do NOT suggest "compelling" for food ("compelling" fits ideas/arguments). Do NOT suggest "beneficial" for a person (sounds impersonal). Do NOT suggest "superb" or "excellent" as generic upgrades for "good" — use topic-specific collocations instead.
"""
        system_prompt += trajectory_line

        user_content = f"""
Candidate's Response ({part}):
"{transcript}"

Identify the topic, then suggest 3-5 context-appropriate vocabulary improvements. Only suggest changes where a genuinely more natural collocation exists for this specific topic.
"""

        try:
            # 1. Local Model Inference
            if self.local_model:
                try:
                    device = self._device
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer(
                        [prompt_text], return_tensors="pt", padding=True, return_attention_mask=True
                    ).to(device)
                    attn_mask = model_inputs.get("attention_mask")
                    if attn_mask is None:
                        attn_mask = torch.ones_like(
                            model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device
                        )
                    
                    # MPS generation workaround (device is str: 'mps'|'cuda'|'cpu')
                    if device == 'mps':
                        pass

                    with torch.inference_mode():
                        generated_ids = self.local_model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=attn_mask,
                            max_new_tokens=400,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                            repetition_penalty=1.1
                        )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    content = _strip_thinking_tags(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
                    
                    # Extract JSON
                    try:
                        start_idx = content.find('[')
                        end_idx = content.rfind(']') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            data = json.loads(json_str)
                            if isinstance(data, list):
                                return data
                    except json.JSONDecodeError:
                        logger.warning("Local model JSON parsing failed for vocab.")
                except Exception as e:
                    logger.error(f"Local Model Vocab Generation failed: {e}")

            # 2. Fallback to API
            if self.client is None and self.api_key and OpenAI:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com",
                    timeout=30.0,
                    max_retries=1
                )

            if self.client is None:
                return []

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            # Handle potential wrapping in a root object
            data = json.loads(content)
            if isinstance(data, dict):
                # Try to find the list
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
                return []
            elif isinstance(data, list):
                return data
            return []

        except Exception as e:
            logger.error(f"Contextual Vocabulary generation failed: {e}")
            return []

    def generate_contextual_word_recommendation(
        self,
        transcript: str,
        part: str = "Part 1",
        user_band: float = 6.0,
        trajectory_context: dict = None,
    ) -> list:
        """
        Generates context-aware word recommendations using DeepSeek API.
        This function specifically targets word-level improvements based on the context.
        Returns: List of dicts {original, alternatives, reason, score, difficulty_range, user_level}
        """
        if not self.is_available():
            return []

        if not transcript or len(transcript.strip()) < 5:
            return []
            
        target_band = min(9.0, user_band + 1.0)
        target_desc = "Band 9 (Expert)" if target_band >= 8.5 else f"Band {target_band}"
        trajectory_context = trajectory_context or {}
        trajectory_line = ""
        targets = trajectory_context.get("target_concepts", [])
        if isinstance(targets, list) and targets:
            trajectory_line = f"\nTRAJECTORY TARGET CONCEPTS: {json.dumps(targets, ensure_ascii=False)}\nPrefer alternatives aligned with these concepts where natural.\n"

        part_register = {
            "Part 1": "everyday spoken collocations and idiomatic phrases suitable for casual conversation",
            "Part 2": "vivid, descriptive narrative language with sensory collocations and storytelling phrases",
            "Part 3": "abstract discussion vocabulary appropriate for spoken debate (not essay-style academic words)",
        }.get(part, "natural spoken collocations")

        system_prompt = f"""You are an expert IELTS Speaking Tutor specializing in Lexical Resource.
Your task is to identify 3-5 specific words in the candidate's response that are below {target_desc} level, and suggest context-aware alternatives appropriate for {target_desc}.

STEP 1 — Identify the topic/domain of the response (e.g. food, travel, technology, relationships, health, education, work, environment).
STEP 2 — For each weak word, suggest alternatives that are natural collocations WITHIN that domain and that a fluent speaker would actually use when talking about this topic.

This is {part} of the IELTS Speaking test. Prefer {part_register}.

OUTPUT FORMAT:
Return a JSON array of objects. Each object must have:
- "original": The exact word from the text.
- "alternatives": An array of 2-3 better alternatives that are natural collocations for the identified topic.
- "reason": Explain why each alternative fits THIS specific topic context better.

EXAMPLES showing how context changes the recommendation:

Text (about travel): "I went to a nice place last year."
Output: [
  {{
    "original": "nice",
    "alternatives": ["stunning", "breathtaking", "charming"],
    "reason": "When describing travel destinations, sensory adjectives like 'stunning' or 'breathtaking' paint a vivid picture, while 'charming' suits smaller or quaint places."
  }},
  {{
    "original": "went to",
    "alternatives": ["visited", "explored", "headed to"],
    "reason": "Travel collocations like 'explored' or the casual spoken phrasal verb 'headed to' are more descriptive than generic 'went to'."
  }}
]

Text (about work): "My job is really hard."
Output: [
  {{
    "original": "really hard",
    "alternatives": ["pretty demanding", "incredibly challenging", "quite intense"],
    "reason": "Work collocations like 'demanding' or 'intense' describe professional difficulty more precisely. Spoken hedges like 'pretty' and 'quite' keep the register natural."
  }}
]

GUIDELINES:
1. Do NOT suggest generic thesaurus synonyms that ignore context. Every alternative must be a natural collocation for the specific topic being discussed.
2. Prefer spoken-register alternatives: idiomatic phrases, phrasal verbs, and collocations used in conversation. Avoid overly literary or essay-style words unless the candidate is in Part 3 discussing abstract ideas.
3. Ensure alternatives are grammatically correct when substituted into the original sentence.
4. Suggest words that are slightly challenging but attainable (i+1 level) for a current Band {user_band} speaker.
5. Consider the surrounding sentence structure — if the candidate uses a verb + noun pattern, suggest a better verb + noun collocation, not just a lone synonym.
6. QUALITY OVER QUANTITY: Suggest only 3-5 improvements where a genuinely more natural option exists. Do NOT mechanically upgrade every word. If the candidate's word is already appropriate for the context, skip it.
7. SOUND NATURAL WHEN SPOKEN: Every alternative must sound natural when said aloud in a conversation. Avoid words that would sound stiff, essay-like, or overly formal when spoken.
8. AVOID THESE MECHANICAL MISTAKES: Do NOT suggest "compelling" for food ("compelling" fits ideas/arguments). Do NOT suggest "beneficial" for a person (sounds impersonal). Do NOT suggest "superb" or "excellent" as generic upgrades for "good" — use topic-specific collocations instead.
"""
        system_prompt += trajectory_line

        user_content = f"""
Candidate's Response ({part}):
"{transcript}"

Identify the topic, then suggest 3-5 context-appropriate word improvements. Only suggest changes where a genuinely more natural collocation exists for this specific topic.
"""

        try:
            content = None
            
            # 1. Local Model Inference
            if self.local_model:
                try:
                    device = self._device
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer(
                        [prompt_text], return_tensors="pt", padding=True, return_attention_mask=True
                    ).to(device)
                    attn_mask = model_inputs.get("attention_mask")
                    if attn_mask is None:
                        attn_mask = torch.ones_like(
                            model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device
                        )
                    
                    with torch.inference_mode():
                        generated_ids = self.local_model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=attn_mask,
                            max_new_tokens=400,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                            repetition_penalty=1.1
                        )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    content = _strip_thinking_tags(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
                except Exception as e:
                    logger.error(f"Local Model Word Rec Generation failed: {e}")
            
            # 2. Fallback to API if local model failed or not available
            if not content:
                if self.client is None and self.api_key and OpenAI:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.deepseek.com",
                        timeout=30.0,
                        max_retries=1
                    )

                if self.client is None:
                    return []

                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content

            # Handle potential wrapping in a root object
            try:
                # Find JSON array in content
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                else:
                    data = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed for word rec.")
                return []

            recommendations = []
            
            raw_list = []
            if isinstance(data, dict):
                # Try to find the list
                for k, v in data.items():
                    if isinstance(v, list):
                        raw_list = v
                        break
            elif isinstance(data, list):
                raw_list = data
                
            for item in raw_list:
                raw_orig = item.get("original")
                orig_str = raw_orig.strip() if isinstance(raw_orig, str) else (str(raw_orig).strip() if isinstance(raw_orig, (int, float)) else "")
                recommendations.append({
                    "original": orig_str,
                    "alternatives": item.get("alternatives"),
                    "reason": item.get("reason"),
                    "score": 85.0,  # Default high score for LLM recommendations
                    "difficulty_range": f"{user_band}-{target_band}",
                    "user_level": user_band
                })
                
            return recommendations

        except Exception as e:
            logger.error(f"Contextual Word Recommendation generation failed: {e}")
            # print(f"ERROR: Contextual Word Recommendation generation failed: {e}") # Force print
            return []

    def generate_grammar_recommendations(self, transcript: str, user_band: float = 6.0) -> list:
        """
        Generates adaptive grammar recommendations based on Zone of Proximal Development (ZPD).
        Returns: List of dicts {original, suggestion, explanation, type}
        """
        if not self.is_available():
            return []

        if not transcript or len(transcript.strip()) < 5:
            return []
            
        # ZPD Logic: Target one step above current level
        target_band = min(9.0, user_band + 1.0)
        
        # tailoring the focus based on proficiency (GRARecom logic)
        focus_area = ""
        if user_band < 5.5:
            focus_area = "Focus on basic accuracy: Subject-verb agreement, correct tense usage, and articles."
        elif user_band < 7.0:
            focus_area = "Focus on range and complexity: Using relative clauses, passive voice, and conditionals."
        else:
            focus_area = "Focus on sophistication: Stylistic inversion, cohesion, and native-like precision."

        system_prompt = f"""You are 'GrammatiCoach', an adaptive grammar tutor.
Your task is to provide 2-3 grammar corrections or enhancements based on the learner's Zone of Proximal Development (Current Band {user_band} -> Target Band {target_band}).

GUIDELINES:
1. {focus_area}
2. Identify specific sentences in the transcript that can be improved.
3. Provide a 'Correction' (or 'Enhancement') and a clear 'Explanation'.
4. Explanation tone: Encouraging for lower bands, analytical for higher bands.

OUTPUT FORMAT:
Return a JSON array of objects:
- "original": The original sentence segment.
- "suggestion": The improved version.
- "explanation": Why this change helps (referencing grammar rules).
- "type": "Correction" (for errors) or "Enhancement" (for better style).
"""

        user_content = f"""
Transcript:
"{transcript}"

Provide grammar feedback.
"""

        try:
            content = None
            
            # 1. Local Model Inference
            if self.local_model:
                try:
                    device = self._device
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer(
                        [prompt_text], return_tensors="pt", padding=True, return_attention_mask=True
                    ).to(device)
                    attn_mask = model_inputs.get("attention_mask")
                    if attn_mask is None:
                        attn_mask = torch.ones_like(
                            model_inputs.input_ids, dtype=torch.long, device=model_inputs.input_ids.device
                        )
                    
                    with torch.inference_mode():
                        generated_ids = self.local_model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=attn_mask,
                            max_new_tokens=400,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                            repetition_penalty=1.1
                        )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    content = _strip_thinking_tags(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
                except Exception as e:
                    logger.error(f"Local Model Grammar Gen failed: {e}")

            # 2. Fallback to API
            if not content:
                if self.client is None and self.api_key and OpenAI:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.deepseek.com",
                        timeout=30.0,
                        max_retries=1
                    )
                
                if self.client is None:
                    return []

                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content

            # Parse JSON
            try:
                # Find JSON array in content
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                else:
                    data = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed for grammar.")
                return []

            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
                return []
            elif isinstance(data, list):
                return data
            return []

        except Exception as e:
            logger.error(f"Grammar recommendation generation failed: {e}")
            return []

# Singleton instance
_evaluator_instance = None

def get_llm_evaluator():
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = DeepSeekEvaluator()
    return _evaluator_instance
