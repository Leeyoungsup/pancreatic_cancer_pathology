from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def discrete_time_survival_nll(
    logits: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    horizon_days: Sequence[float] | torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Discrete-time survival negative log-likelihood.

    The model outputs interval hazards for horizons such as 6, 12, 18, 24 months.
    A death inside an observed interval contributes survival through previous
    intervals and event probability in the event interval. Censored samples
    contribute survival terms for intervals known to have been survived.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    times = times.reshape(-1).to(logits.device).float()
    events = events.reshape(-1).to(logits.device).long()
    horizons = torch.as_tensor(horizon_days, dtype=torch.float32, device=logits.device)

    hazards = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    log_hazard = torch.log(hazards)
    log_survival = torch.log1p(-hazards)

    losses = []
    for i in range(logits.shape[0]):
        time_i = times[i]
        event_i = events[i]
        if torch.isnan(time_i):
            continue

        if event_i.item() == 1 and time_i <= horizons[-1]:
            event_idx = int(torch.searchsorted(horizons, time_i, right=False).item())
            terms = []
            if event_idx > 0:
                terms.append(log_survival[i, :event_idx].sum())
            terms.append(log_hazard[i, event_idx])
            losses.append(-torch.stack(terms).sum())
        else:
            known_survived = horizons <= time_i
            if known_survived.any():
                losses.append(-log_survival[i, known_survived].sum())

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def hazards_to_cumulative_risk(logits: torch.Tensor) -> torch.Tensor:
    """Convert interval hazard logits to cumulative risk at each horizon."""
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=-1)
    return 1.0 - survival


def cox_ph_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    Higher risk score indicates shorter survival. The loss is computed within
    the provided case batch, so the batch should contain multiple cases.
    """
    risk_scores = risk_scores.reshape(-1)
    times = times.reshape(-1).to(risk_scores.device).float()
    events = events.reshape(-1).to(risk_scores.device).float()

    valid = torch.isfinite(times)
    risk_scores = risk_scores[valid]
    times = times[valid]
    events = events[valid]
    if risk_scores.numel() == 0 or events.sum() == 0:
        return risk_scores.sum() * 0.0

    order = torch.argsort(times, descending=True)
    sorted_risk = risk_scores[order]
    sorted_events = events[order]
    log_cumsum_exp = torch.logcumsumexp(sorted_risk, dim=0)
    event_terms = (sorted_risk - log_cumsum_exp) * sorted_events
    return -event_terms.sum() / sorted_events.sum().clamp_min(1.0)


def harrell_c_index(times: Sequence[float], events: Sequence[int], risks: Sequence[float]) -> float:
    """Harrell's concordance index using higher risk = shorter survival."""
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    risks_arr = np.asarray(risks, dtype=float)

    concordant = 0.0
    comparable = 0
    n = len(times_arr)
    for i in range(n):
        if not np.isfinite(times_arr[i]) or events_arr[i] != 1:
            continue
        for j in range(n):
            if i == j or not np.isfinite(times_arr[j]):
                continue
            if times_arr[i] < times_arr[j]:
                comparable += 1
                if risks_arr[i] > risks_arr[j]:
                    concordant += 1.0
                elif risks_arr[i] == risks_arr[j]:
                    concordant += 0.5
    if comparable == 0:
        return float("nan")
    return concordant / comparable


def horizon_auc_metrics(
    labels: np.ndarray,
    masks: np.ndarray,
    risks: np.ndarray,
    horizon_names: Sequence[str],
) -> dict[str, float]:
    """Compute horizon-wise AUROC/AUPRC under known-label masks."""
    metrics: dict[str, float] = {}
    aurocs = []
    auprcs = []
    for h, name in enumerate(horizon_names):
        known = masks[:, h].astype(bool)
        y_true = labels[known, h].astype(int)
        y_score = risks[known, h].astype(float)

        metrics[f"{name}_known"] = int(known.sum())
        metrics[f"{name}_positive"] = int(y_true.sum()) if len(y_true) else 0
        if len(np.unique(y_true)) < 2:
            metrics[f"{name}_auroc"] = float("nan")
            metrics[f"{name}_auprc"] = float("nan")
            continue

        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        metrics[f"{name}_auroc"] = auroc
        metrics[f"{name}_auprc"] = auprc
        aurocs.append(auroc)
        auprcs.append(auprc)

    metrics["mean_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    metrics["mean_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    return metrics
