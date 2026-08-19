"""
可插拔 LLM 客户端（诚实版）
====================
- 供应商：DeepSeek API（deepseek-chat / deepseek-reasoner）、本地 qwen（llama.cpp server，端口 8001）
- 配置：用户自建 src/llm/llm-config.local.json，代码只读不打印 key
- 输出解析：要求模型输出 JSON {probs:[pH,pD,pA], reasoning:"..."}，解析失败则标记失败
  并返回 None（上层决定兜底策略），绝不静默编造概率
- 成本追踪：记录 token 用量与估算成本

llm-config.local.json 格式（用户自建，勿提交 git）：
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
                "缺少 llm-config.local.json（用户自建，含 API key，勿提交 git）")
        self.provider = provider or self.config.get("default_provider", "deepseek")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---------- 底层调用 ----------
    def _call(self, messages, temperature=0.3, max_tokens=1200, n=1):
        """返回 list[dict]，每个 dict 含 content；失败抛异常。"""
        cfg = self.config[self.provider]
        if self.provider == "local":
            import urllib.request
            model_id = cfg.get("model")
            if not model_id:
                # 自动解析服务端已加载模型 ID（本地服务，无隐私内容）
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
        else:  # deepseek (OpenAI 兼容)
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
                out.append({"content": c["message"]["content"]})
            usage = data.get("usage", {})
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            return out

    # ---------- 概率解析 ----------
    @staticmethod
    def parse_probs(text):
        """从模型输出中提取 [pH, pD, pA]；失败返回 None（绝不静默编造）。"""
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

    # ---------- 领域增强预测 ----------
    def predict_match(self, match_prompt, n_samples=3, temperature=0.3):
        """
        对一场比赛做 n_samples 次推理，返回:
        (probs_list, ok_count, reasons)
        - probs_list: 每次的 [pH,pD,pA]（解析失败的样本为 None）
        - ok_count: 成功解析的样本数
        - reasons: 成功样本的 reasoning 文本
        """
        from llm.prompts import build_messages
        messages = build_messages(match_prompt)
        probs_list, reasons = [], []
        ok = 0
        for _ in range(n_samples):
            try:
                out = self._call(messages, temperature=temperature)
                content = out[0]["content"]
                p = self.parse_probs(content)
                if p is None:
                    probs_list.append(None)
                    continue
                probs_list.append(p)
                reasons.append(content)
                ok += 1
            except Exception as e:
                probs_list.append(None)
                print(f"  [llm] 调用失败: {e}")
        return probs_list, ok, reasons

    def estimate_cost(self, price_per_m_input=0.14, price_per_m_output=0.28):
        """DeepSeek-chat 参考价：$0.14/M input, $0.28/M output（美元/百万 token）。本地模型返回 0。"""
        if self.provider == "local":
            return 0.0
        return (self.total_prompt_tokens / 1e6 * price_per_m_input
                + self.total_completion_tokens / 1e6 * price_per_m_output)


def np_isfinite(x):
    try:
        import math
        return math.isfinite(x)
    except Exception:
        return True
