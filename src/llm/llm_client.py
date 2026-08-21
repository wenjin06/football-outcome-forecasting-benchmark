"""
Pluggable LLM client
====================
- Providers: DeepSeek API (deepseek-chat / deepseek-reasoner), local qwen
  (llama.cpp server, port 8001)
- Configuration: user-created src/llm/llm-config.local.json; the code only reads
  it and never prints the key
- Output parsing: the model is asked to output JSON {probs:[pH,pD,pA], reasoning:"..."};
  on parse failure the sample is marked as failed and None is returned (the caller
  decides the fallback policy); probabilities are never silently fabricated
- Cost tracking: records token usage and estimated cost

llm-config.local.json format (user-created; do not commit to git):
{
  "default_provider": "deepseek",
  "deepseek": { "api_key": "sk-...", "base_url": "https://api.deepseek.com", "model": "deepseek-chat" },
  "local":    { "base_url": "http://127.0.0.1:8001", "model": "qwen2.5-coder-7b" }
}
"""
import json
import os
import re
import time

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "llm-config.local.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class LLMClient:
    def __init__(self, config=None, provider=None):
        self.config = config or load_config()
        if not self.config:
            raise RuntimeError(
                "Missing llm-config.local.json (user-created, contains API key, do not commit to git)")
        self.provider = provider or self.config.get("default_provider", "deepseek")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---------- Low-level calls ----------
    def _call(self, messages, temperature=0.3, max_tokens=1200, n=1):
        """Returns a list of dicts, each with a content field; raises on failure."""
        cfg = self.config[self.provider]
        if self.provider == "local":
            import urllib.request
            model_id = cfg.get("model")
            if not model_id:
                # Auto-resolve the model ID loaded by the server (local service, no private content)
                with urllib.request.urlopen(
                        cfg["base_url"].rstrip("/") + "/v1/models", timeout=30) as resp:
                    model_id = json.loads(resp.read().decode("utf-8"))["data"][0]["id"]
            req = urllib.request.Request(
                cfg["base_url"].rstrip("/") + "/v1/chat/completions",
                data=json.dumps({
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = []
            for c in data.get("choices", []):
                out.append({"content": c["message"]["content"]})
            usage = data.get("usage", {})
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            return out
        else:  # deepseek (OpenAI-compatible)
            import urllib.request
            payload = {
                "model": cfg["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            }
            req = urllib.request.Request(
                cfg["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + cfg["api_key"],
                })
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = []
            for c in data.get("choices", []):
                msg = c.get("message", {})
                out.append({"content": msg.get("content") or "",
                            "reasoning": msg.get("reasoning_content") or ""})
            usage = data.get("usage", {})
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            return out

    # ---------- Probability parsing ----------
    @staticmethod
    def parse_probs(text):
        """Extract [pH, pD, pA] from the model output; returns None on failure (never silently fabricates)."""
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        probs = obj.get("probs") or obj.get("probabilities")
        if not isinstance(probs, (list, dict)):
            return None
        if isinstance(probs, dict):
            probs = [probs.get("home", probs.get("H")),
                     probs.get("draw", probs.get("D")),
                     probs.get("away", probs.get("A"))]
        probs = [float(x) for x in probs]
        if len(probs) != 3:
            return None
        s = sum(probs)
        if s <= 0 or not all(np_isfinite(p) for p in probs):
            return None
        return [p / s for p in probs]

    # ---------- Domain-augmented prediction ----------
    def predict_match(self, match_prompt, n_samples=3, temperature=0.3,
                      feature_mode="full"):
        """
        Run n_samples inferences for one match; returns:
        (probs_list, ok_count, reasons)
        - probs_list: [pH,pD,pA] per sample (None for samples that failed parsing)
        - ok_count: number of samples parsed successfully
        - reasons: reasoning text of the successful samples
        feature_mode: see prompts.build_feature_card (LLM input ablation)
        """
        from llm.prompts import build_messages
        messages = build_messages(match_prompt, mode=feature_mode)
        probs_list, reasons = [], []
        ok = 0
        for _ in range(n_samples):
            try:
                out = self._call(messages, temperature=temperature,
                                 max_tokens=4000)
                content = out[0].get("content") or ""
                p = self.parse_probs(content)
                if p is None and out[0].get("reasoning"):
                    # deepseek-reasoner puts reasoning in reasoning_content;
                    # the final answer may be empty or truncated: try parsing JSON from the reasoning
                    p = self.parse_probs(out[0]["reasoning"])
                if p is None:
                    probs_list.append(None)
                    continue
                probs_list.append(p)
                reasons.append(content or out[0].get("reasoning", ""))
                ok += 1
            except Exception as e:
                probs_list.append(None)
                print(f"  [llm] call failed: {e}")
        return probs_list, ok, reasons

    def estimate_cost(self):
        """Estimate cost from the actual model prices (USD per million tokens). Local models return 0.
        deepseek-chat: $0.14/M in, $0.28/M out; deepseek-reasoner: $0.55/M in, $2.19/M out."""
        if self.provider == "local":
            return 0.0
        model = (self.config.get(self.provider, {}) or {}).get("model", "")
        if "reasoner" in model:
            p_in, p_out = 0.55, 2.19
        else:
            p_in, p_out = 0.14, 0.28
        return (self.total_prompt_tokens / 1e6 * p_in
                + self.total_completion_tokens / 1e6 * p_out)


def np_isfinite(x):
    try:
        import math
        return math.isfinite(x)
    except Exception:
        return True
