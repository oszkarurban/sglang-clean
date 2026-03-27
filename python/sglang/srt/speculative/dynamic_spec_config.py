# """
# python/sglang/srt/speculative/dynamic_spec_config.py

# CUDA graph capture safety
# --------------------------
# Any .item() or .cpu() call inside a CUDA graph capture window raises:
#   "CUDA error: operation not permitted when stream is capturing"
# Every collect_* method calls _is_capturing() first and returns immediately
# if inside a capture. Signal collection only runs during real inference.
# """

# from __future__ import annotations

# import json
# import logging
# import math
# import os
# from dataclasses import dataclass, field
# from typing import List, Optional, Tuple

# import torch

# logger = logging.getLogger(__name__)


# def _is_capturing() -> bool:
#     try:
#         return torch.cuda.is_current_stream_capturing()
#     except Exception:
#         return False


# @dataclass
# class DynamicSpecConfig:
#     # ── master switch ────────────────────────────────────────────────────────
#     enabled: bool = False

#     # ── vocab size (required) ────────────────────────────────────────────────
#     # Pass vocab_size=model_config.vocab_size from EAGLEWorker.__init__
#     # max_entropy_ref = log(vocab_size) is computed in __post_init__
#     vocab_size: int = 128000
#     max_entropy_ref: float = field(init=False, repr=True)

#     # ── tree shape policy ─────────────────────────────────────────────────────
#     # Bidirectional 3-tier policy: steps and dtn are independent variables.
#     # All values set by eagle_worker.py at init from server_args.
#     baseline_steps: Optional[int] = None   # ← --speculative-num-steps-startpoint
#     baseline_dtn:   Optional[int] = None   # ← --speculative-num-draft-tokens-startpoint
#     max_steps:      Optional[int] = None   # ← --speculative-num-steps (server max)
#     max_dtn:        Optional[int] = None   # ← --speculative-num-draft-tokens (server max)
#     topk:           Optional[int] = None   # ← --speculative-eagle-topk (fixed at runtime)

#     # ── policy thresholds ─────────────────────────────────────────────────────
#     # Set AFTER running probe_signal_ranges.py.
#     high_conf_threshold: float = 0.35
#     low_conf_threshold:  float = 0.65

#     # ── which signals to collect ─────────────────────────────────────────────
#     draft_entropy:       bool = True
#     top1_prob:           bool = True
#     top1_minus_top2:     bool = True
#     hidden_norm:         bool = True
#     path_score:          bool = True
#     target_entropy:      bool = True
#     entropy_gap:         bool = True
#     rolling_accept_rate: bool = True
#     rolling_window:      int  = 8

#     # ── signal logging ────────────────────────────────────────────────────────
#     # When set, decide() appends one JSONL line per verify call to this file.
#     # Pass via --dynamic-spec-config '{"signal_log_path": "probe_logs/run.jsonl", ...}'
#     signal_log_path: Optional[str] = None

#     # ── combination weights — equal until empirically tuned ──────────────────
#     weight_draft_entropy:       float = 1.0
#     weight_top1_prob:           float = 1.0
#     weight_top1_minus_top2:     float = 1.0
#     weight_hidden_norm:         float = 1.0
#     weight_path_score:          float = 1.0
#     weight_target_entropy:      float = 1.0
#     weight_entropy_gap:         float = 1.0
#     weight_rolling_accept_rate: float = 1.0

#     # ── per-step state (reset each verify step) ───────────────────────────────
#     _draft_entropies:  List[float] = field(default_factory=list, repr=False)
#     _top1_probs:       List[float] = field(default_factory=list, repr=False)
#     _top1_margins:     List[float] = field(default_factory=list, repr=False)
#     _hidden_norms:     List[float] = field(default_factory=list, repr=False)
#     _path_score:       Optional[float] = field(default=None, repr=False)
#     _target_entropy:   Optional[float] = field(default=None, repr=False)
#     _step_accept_rate: Optional[float] = field(default=None, repr=False)

#     def __post_init__(self):
#         self.max_entropy_ref = math.log(self.vocab_size)

#     # ── public API ────────────────────────────────────────────────────────────

#     def reset(self):
#         """Call at the start of each verify step to clear per-step state."""
#         self._draft_entropies.clear()
#         self._top1_probs.clear()
#         self._top1_margins.clear()
#         self._hidden_norms.clear()
#         self._path_score = None
#         self._target_entropy = None
#         self._step_accept_rate = None

#     def collect_draft_signals(
#         self,
#         probs: torch.Tensor,          # (bs*topk, vocab) after softmax
#         hidden_states: torch.Tensor,  # (bs*topk, hidden)
#         step_i: int,
#     ):
#         """
#         Call in draft_forward() after each softmax, before fast_topk.
#         Returns immediately during CUDA graph capture — safe to call always.
#         """
#         if not self.enabled:
#             return
#         if _is_capturing():          # ← CUDA graph capture guard
#             return
#         with torch.no_grad():
#             if self.draft_entropy or self.entropy_gap:
#                 ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
#                 self._draft_entropies.append(ent)
#             if self.top1_prob:
#                 self._top1_probs.append(probs.max(dim=-1).values.mean().item())
#             if self.top1_minus_top2 and probs.shape[-1] >= 2:
#                 top2 = probs.topk(2, dim=-1).values
#                 self._top1_margins.append((top2[:, 0] - top2[:, 1]).mean().item())
#             if self.hidden_norm:
#                 self._hidden_norms.append(hidden_states.norm(dim=-1).mean().item())

#     def collect_path_score(self, score_list_flat: torch.Tensor):
#         """
#         Call in organize_draft_results() after flattening, before topk.
#         Returns immediately during CUDA graph capture — safe to call always.
#         """
#         if not self.enabled or not self.path_score:
#             return
#         if _is_capturing():          # ← CUDA graph capture guard
#             return
#         with torch.no_grad():
#             self._path_score = score_list_flat.max(dim=-1).values.mean().item()

#     def collect_target_signals(
#         self,
#         logits: torch.Tensor,  # (bs * draft_token_num, vocab)
#         bs: int,
#         draft_token_num: int,
#     ):
#         """
#         Call in verify() after target forward, before accept/reject.
#         Returns immediately during CUDA graph capture — safe to call always.
#         """
#         if not self.enabled:
#             return
#         if not (self.target_entropy or self.entropy_gap):
#             return
#         if _is_capturing():          # ← CUDA graph capture guard
#             return
#         with torch.no_grad():
#             probs = torch.softmax(logits.float(), dim=-1)
#             self._target_entropy = (
#                 -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
#             )

#     def collect_accept_signal(
#         self,
#         accept_index_row: List[int],
#         draft_token_num: int,
#     ):
#         """
#         Call in verify() inside the per-req loop after accept_index_row.
#         Returns immediately during CUDA graph capture — safe to call always.
#         """
#         if not self.enabled or not self.rolling_accept_rate:
#             return
#         if _is_capturing():          # ← CUDA graph capture guard
#             return
#         n_acc = max(0, sum(1 for idx in accept_index_row if idx != -1) - 1)
#         possible = draft_token_num - 1
#         self._step_accept_rate = n_acc / possible if possible > 0 else 0.0

#     def decide(self, req) -> Tuple[int, int]:
#         """
#         Combine enabled signals → return (spec_steps, num_draft_tokens).

#         Args:
#             req: Req object (for rolling accept rate histogram)
#         """
#         if not self.enabled:
#             return self.max_steps, self.max_dtn

#         score = self._compute_combined_score(req)
#         steps, dtn = self._policy(score)

#         logger.debug(
#             f"DynamicSpec: score={score:.3f} → steps={steps} dtn={dtn} | "
#             f"draft_ent={self._mean(self._draft_entropies):.3f} "
#             f"path={self._path_score} "
#             f"tgt_ent={self._target_entropy} "
#             f"acc={self._step_accept_rate}"
#         )

#         if self.signal_log_path:
#             log_dir = os.path.dirname(self.signal_log_path)
#             if log_dir:
#                 os.makedirs(log_dir, exist_ok=True)
#             record = {"score": score, "chosen_steps": steps, "chosen_dtn": dtn}
#             record.update(self.summary())
#             record["rolling_accept_rate_value"] = self._get_rolling_accept_rate(req)
#             with open(self.signal_log_path, "a") as _f:
#                 _f.write(json.dumps(record) + "\n")

#         return steps, dtn

#     def summary(self) -> dict:
#         """Return raw signal values for this step — used in logging/debug scripts."""
#         return {
#             "draft_entropies":  list(self._draft_entropies),
#             "top1_probs":       list(self._top1_probs),
#             "top1_margins":     list(self._top1_margins),
#             "hidden_norms":     list(self._hidden_norms),
#             "path_score":       self._path_score,
#             "target_entropy":   self._target_entropy,
#             "step_accept_rate": self._step_accept_rate,
#             "max_entropy_ref":  self.max_entropy_ref,
#         }

#     # ── internal ──────────────────────────────────────────────────────────────

#     @staticmethod
#     def _mean(lst: List[float]) -> float:
#         return sum(lst) / len(lst) if lst else 0.0

#     def _compute_combined_score(self, req) -> float:
#         """Combined uncertainty score in [0, 1]. 0=confident, 1=uncertain."""
#         total_w = 0.0
#         total_s = 0.0

#         def add(val: float, w: float):
#             nonlocal total_w, total_s
#             total_w += w
#             total_s += w * max(0.0, min(1.0, val))

#         if self.draft_entropy and self._draft_entropies:
#             add(self._mean(self._draft_entropies) / self.max_entropy_ref,
#                 self.weight_draft_entropy)

#         if self.top1_prob and self._top1_probs:
#             add(1.0 - self._mean(self._top1_probs), self.weight_top1_prob)

#         if self.top1_minus_top2 and self._top1_margins:
#             add(1.0 - min(self._mean(self._top1_margins), 1.0),
#                 self.weight_top1_minus_top2)

#         if self.hidden_norm and self._hidden_norms:
#             add(min(self._mean(self._hidden_norms) / 100.0, 1.0),
#                 self.weight_hidden_norm)

#         if self.path_score and self._path_score is not None:
#             add(1.0 - self._path_score, self.weight_path_score)

#         if self.target_entropy and self._target_entropy is not None:
#             add(self._target_entropy / self.max_entropy_ref,
#                 self.weight_target_entropy)

#         if (self.entropy_gap
#                 and self._draft_entropies
#                 and self._target_entropy is not None):
#             gap = self._mean(self._draft_entropies) - self._target_entropy
#             # normalised = (gap / self.max_entropy_ref + 1.0) / 2.0
#             normalised = (-gap / self.max_entropy_ref + 1.0) / 2.0
#             add(normalised, self.weight_entropy_gap)

#         if self.rolling_accept_rate:
#             rate = self._get_rolling_accept_rate(req)
#             if rate is not None:
#                 add(1.0 - rate, self.weight_rolling_accept_rate)

#         return total_s / total_w if total_w > 0.0 else 0.0

#     def _get_rolling_accept_rate(self, req) -> Optional[float]:
#         hist = getattr(req, "spec_acceptance_histogram", None)
#         if not hist:
#             return None
#         total_steps = sum(hist)
#         if total_steps == 0:
#             return None
#         max_accepted = len(hist) - 1
#         if max_accepted == 0:
#             return 0.0
#         total_accepted = sum(i * hist[i] for i in range(len(hist)))
#         return total_accepted / (total_steps * max_accepted)

#     def _policy(self, score: float) -> Tuple[int, int]:
#         """
#         Bidirectional 3-tier policy. steps and dtn are independent.

#         High confidence (score < high_conf_threshold):
#             → max_steps, max_dtn  (CUDA graph eligible — matches captured shape)
#         Neutral (between thresholds):
#             → baseline_steps, baseline_dtn  (eager fallback if != max)
#         Low confidence (score >= low_conf_threshold):
#             → 1 step, topk + 1 dtn  (minimal speculation, eager fallback)
#         """
#         if score < self.high_conf_threshold:
#             steps = self.max_steps
#             dtn = self.max_dtn
#         elif score < self.low_conf_threshold:
#             steps = self.baseline_steps
#             dtn = self.baseline_dtn
#         else:
#             steps = 1
#             dtn = self.topk + 1

#         return steps, dtn