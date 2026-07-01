# M1 학습 및 validation
# patient-level Cox partial likelihood loss, C-index, KM curve, log-rank, hazard ratio, checkpoint, scheduler

from PIL import Image
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def load_tile_tensor_batch(tile_paths: list[str], transform, device: torch.device) -> torch.Tensor:
    images = []
    use_cache = bool(globals().get("TILE_IMAGE_CACHE", {}))
    for path in tile_paths:
        if use_cache and path in TILE_IMAGE_CACHE:
            image = Image.fromarray(TILE_IMAGE_CACHE[path])
        else:
            with Image.open(path) as image_file:
                image = image_file.convert("RGB")
        images.append(transform(image))
    return torch.stack(images, dim=0).to(device, non_blocking=True)


def prepare_slide_batch(sample: dict, training: bool) -> dict:
    if bool(globals().get("TILE_IMAGE_CACHE", {})):
        transform = get_train_cached_patch_transform() if training else get_eval_cached_patch_transform()
    else:
        transform = get_train_patch_transform() if training else get_eval_patch_transform()
    selected_paths, selected_coords, selected_indices = sample_tiles(
        sample["tile_paths"],
        sample["coords"],
        max_tiles=MAX_TILES_PER_SLIDE,
        training=training,
    )
    tile_images = load_tile_tensor_batch(selected_paths, transform=transform, device=device)
    return {
        "tile_images": tile_images,
        "coords": selected_coords.to(device, non_blocking=True),
        "os_time_days": sample["os_time_days"].reshape(1).to(device, non_blocking=True).float(),
        "os_event": sample["os_event"].reshape(1).to(device, non_blocking=True).long(),
        "dataset": sample["dataset"],
        "case_id": sample["case_id"],
        "slide_uid": sample.get("slide_uid", sample["case_id"]),
        "n_tiles": len(selected_paths),
    }


def logrank_test(times: np.ndarray, events: np.ndarray, group: np.ndarray) -> dict[str, float]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    group = np.asarray(group, dtype=int)
    event_times = np.sort(np.unique(times[(events == 1) & np.isfinite(times)]))
    observed_high = 0.0
    expected_high = 0.0
    variance_high = 0.0
    for t in event_times:
        at_risk = times >= t
        event_at_t = (times == t) & (events == 1)
        n = float(at_risk.sum())
        if n <= 1:
            continue
        n_high = float((at_risk & (group == 1)).sum())
        d = float(event_at_t.sum())
        d_high = float((event_at_t & (group == 1)).sum())
        expected = d * n_high / n
        var = n_high * (n - n_high) * d * (n - d) / (n * n * max(n - 1.0, 1.0))
        observed_high += d_high
        expected_high += expected
        variance_high += var
    chi2_stat = (observed_high - expected_high) ** 2 / max(variance_high, 1e-12)
    p_value = float(math.erfc(math.sqrt(max(chi2_stat, 0.0) / 2.0)))
    return {
        "observed_high": observed_high,
        "expected_high": expected_high,
        "chi2": chi2_stat,
        "p_value": p_value,
    }


def cox_binary_hr(times: np.ndarray, events: np.ndarray, group: np.ndarray, max_iter: int = 50) -> dict[str, float]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    x = np.asarray(group, dtype=float)
    valid = np.isfinite(times)
    times, events, x = times[valid], events[valid], x[valid]
    if len(np.unique(x)) < 2 or events.sum() == 0:
        return {"hr": float("nan"), "hr_ci_low": float("nan"), "hr_ci_high": float("nan"), "beta": float("nan"), "se": float("nan")}
    beta = 0.0
    info = np.nan
    for _ in range(max_iter):
        score = 0.0
        info = 0.0
        for t, e, xi in zip(times, events, x):
            if e != 1:
                continue
            risk = times >= t
            w = np.exp(np.clip(beta * x[risk], -50, 50))
            sw = w.sum()
            mean_x = (w * x[risk]).sum() / sw
            mean_x2 = (w * x[risk] * x[risk]).sum() / sw
            score += xi - mean_x
            info += mean_x2 - mean_x * mean_x
        if info <= 1e-12:
            break
        step = score / info
        beta += step
        if abs(step) < 1e-6:
            break
    se = float(np.sqrt(1.0 / max(info, 1e-12)))
    hr = float(np.exp(beta))
    return {
        "hr": hr,
        "hr_ci_low": float(np.exp(beta - 1.96 * se)),
        "hr_ci_high": float(np.exp(beta + 1.96 * se)),
        "beta": float(beta),
        "se": se,
    }


def km_curve(times: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times)
    times, events = times[order], events[order]
    surv = 1.0
    rows = [{"time": 0.0, "survival": 1.0}]
    for t in np.unique(times[events == 1]):
        at_risk = np.sum(times >= t)
        deaths = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            surv *= 1.0 - deaths / at_risk
            rows.append({"time": float(t), "survival": float(surv)})
    return pd.DataFrame(rows)


def compute_epoch_metrics(risk_scores: list[float], times: list[float], events: list[int], datasets: list[str], case_ids: list[str]) -> dict:
    risk_array = np.asarray(risk_scores, dtype=float)
    time_array = np.asarray(times, dtype=float)
    event_array = np.asarray(events, dtype=int)
    metrics = {
        "c_index": harrell_c_index(time_array, event_array, risk_array) if len(risk_array) else float("nan"),
        "risk_mean": float(np.mean(risk_array)) if len(risk_array) else float("nan"),
        "risk_std": float(np.std(risk_array)) if len(risk_array) else float("nan"),
    }
    if len(risk_array) and np.isfinite(risk_array).all():
        cutoff = float(np.median(risk_array))
        group = (risk_array >= cutoff).astype(int)
        lr = logrank_test(time_array, event_array, group)
        hr = cox_binary_hr(time_array, event_array, group)
        metrics.update({
            "risk_cutoff_median": cutoff,
            "logrank_p_value": lr["p_value"],
            "logrank_chi2": lr["chi2"],
            "hr_high_vs_low": hr["hr"],
            "hr_ci_low": hr["hr_ci_low"],
            "hr_ci_high": hr["hr_ci_high"],
        })
    else:
        metrics.update({"risk_cutoff_median": float("nan"), "logrank_p_value": float("nan"), "logrank_chi2": float("nan"), "hr_high_vs_low": float("nan"), "hr_ci_low": float("nan"), "hr_ci_high": float("nan")})
    metrics["prediction_df"] = pd.DataFrame({
        "dataset": datasets,
        "case_id": case_ids,
        "os_time_days": time_array,
        "os_event": event_array,
        "risk_score": risk_array,
    })
    if len(risk_array):
        metrics["prediction_df"]["risk_group"] = np.where(risk_array >= metrics["risk_cutoff_median"], "High risk", "Low risk")
    return metrics


def metrics_for_checkpoint(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "prediction_df"}


def finalize_case_batch(logits_list, times_list, events_list, training: bool) -> torch.Tensor:
    logits_tensor = torch.cat(logits_list, dim=0)
    times_tensor = torch.cat(times_list, dim=0)
    events_tensor = torch.cat(events_list, dim=0)
    loss = m1_loss_fn(logits_tensor, times_tensor, events_tensor)
    if training:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
    return loss


def run_one_epoch(dataset, training: bool, epoch: int) -> dict:
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    all_risks = []
    all_times = []
    all_events = []
    all_datasets = []
    all_case_ids = []
    logits_list = []
    times_list = []
    events_list = []

    progress = tqdm(range(len(dataset)), desc=f"Epoch {epoch:03d} {'train' if training else 'valid'}", leave=False)
    for idx in progress:
        sample = dataset[idx]
        with torch.set_grad_enabled(training):
            batch = prepare_slide_batch(sample, training=training)
            outputs = model(batch["tile_images"], batch["coords"])
            risk = outputs["logits"].reshape(-1)
            logits_list.append(risk)
            times_list.append(batch["os_time_days"])
            events_list.append(batch["os_event"].float())

        all_risks.append(float(risk.detach().cpu().item()))
        all_times.append(float(batch["os_time_days"].detach().cpu().item()))
        all_events.append(int(batch["os_event"].detach().cpu().item()))
        all_datasets.append(batch["dataset"])
        all_case_ids.append(batch["case_id"])

        is_batch_ready = len(logits_list) >= CASE_BATCH_SIZE or idx == len(dataset) - 1
        if is_batch_ready:
            loss = finalize_case_batch(logits_list, times_list, events_list, training=training)
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            logits_list, times_list, events_list = [], [], []
            running_loss = total_loss / max(total_batches, 1)
            running_metrics = compute_epoch_metrics(all_risks, all_times, all_events, all_datasets, all_case_ids)
            progress.set_postfix({
                "avg_loss": f"{running_loss:.4f}",
                "c_index": "nan" if np.isnan(running_metrics["c_index"]) else f"{running_metrics['c_index']:.3f}",
                "hr": "nan" if np.isnan(running_metrics["hr_high_vs_low"]) else f"{running_metrics['hr_high_vs_low']:.2f}",
                "p": "nan" if np.isnan(running_metrics["logrank_p_value"]) else f"{running_metrics['logrank_p_value']:.3g}",
            })
            del loss

        del batch, outputs, risk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {"loss": total_loss / max(total_batches, 1)}
    metrics.update(compute_epoch_metrics(all_risks, all_times, all_events, all_datasets, all_case_ids))
    return metrics


def get_monitor_score(metrics: dict, prefix: str = "valid") -> float:
    key = MONITOR_METRIC.replace(f"{prefix}_", "")
    return float(metrics.get(key, np.nan))


def save_checkpoint(path: Path, epoch: int, metrics: dict, is_best: bool) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": {"train": metrics_for_checkpoint(metrics["train"]), "valid": metrics_for_checkpoint(metrics["valid"]), "monitor_score": metrics["monitor_score"]},
        "training_config": training_config,
        "is_best": is_best,
    }, path)


def plot_km_by_risk(pred_df: pd.DataFrame, title: str) -> None:
    if pred_df.empty or pred_df["risk_group"].nunique() < 2:
        return
    plt.figure(figsize=(6, 5))
    for group_name, part in pred_df.groupby("risk_group"):
        km = km_curve(part["os_time_days"].to_numpy(float), part["os_event"].to_numpy(int))
        plt.step(km["time"] / 30.4375, km["survival"], where="post", label=f"{group_name} (n={len(part)})")
    plt.title(title)
    plt.xlabel("Months")
    plt.ylabel("Overall survival probability")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()


def plot_training_history(history: list[dict], valid_pred_df: pd.DataFrame | None = None) -> None:
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0, 0].plot(hist_df["epoch"], hist_df["valid_loss"], label="valid")
    axes[0, 0].set_title("Cox Partial Likelihood Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(hist_df["epoch"], hist_df["train_c_index"], label="train")
    axes[0, 1].plot(hist_df["epoch"], hist_df["valid_c_index"], label="valid")
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("C-index")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(hist_df["epoch"], hist_df["valid_hr_high_vs_low"], label="valid HR")
    axes[1, 0].set_title("Valid HR: high vs low risk")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(hist_df["epoch"], hist_df["lr"], label="lr")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
    display(hist_df.tail(10))
    if valid_pred_df is not None:
        last = hist_df.iloc[-1]
        plot_km_by_risk(
            valid_pred_df,
            title=f"Validation KM by median risk | epoch={int(last['epoch'])}, C-index={last['valid_c_index']:.3f}, p={last['valid_logrank_p']:.3g}",
        )


history = []
best_score = -np.inf if MONITOR_MODE == "max" else np.inf
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    train_metrics = run_one_epoch(train_dataset, training=True, epoch=epoch)
    valid_metrics = run_one_epoch(valid_dataset, training=False, epoch=epoch)

    monitor_score = get_monitor_score(valid_metrics, prefix="valid")
    scheduler.step(monitor_score)
    current_lr = optimizer.param_groups[0]["lr"]

    improved = monitor_score > best_score + MIN_DELTA if MONITOR_MODE == "max" else monitor_score < best_score - MIN_DELTA
    if improved:
        best_score = monitor_score
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "valid_loss": valid_metrics["loss"],
        "train_c_index": train_metrics["c_index"],
        "valid_c_index": valid_metrics["c_index"],
        "train_logrank_p": train_metrics["logrank_p_value"],
        "valid_logrank_p": valid_metrics["logrank_p_value"],
        "train_hr_high_vs_low": train_metrics["hr_high_vs_low"],
        "train_hr_ci_low": train_metrics["hr_ci_low"],
        "train_hr_ci_high": train_metrics["hr_ci_high"],
        "valid_hr_high_vs_low": valid_metrics["hr_high_vs_low"],
        "valid_hr_ci_low": valid_metrics["hr_ci_low"],
        "valid_hr_ci_high": valid_metrics["hr_ci_high"],
        "train_risk_mean": train_metrics["risk_mean"],
        "valid_risk_mean": valid_metrics["risk_mean"],
        "train_risk_std": train_metrics["risk_std"],
        "valid_risk_std": valid_metrics["risk_std"],
        "lr": current_lr,
    }
    history.append(row)

    log_df = pd.DataFrame(history)
    log_df.to_csv(TRAIN_LOG_PATH, index=False)

    checkpoint_metrics = {"train": train_metrics, "valid": valid_metrics, "monitor_score": monitor_score}
    save_checkpoint(LAST_CHECKPOINT_PATH, epoch, checkpoint_metrics, is_best=False)
    if improved:
        save_checkpoint(BEST_CHECKPOINT_PATH, epoch, checkpoint_metrics, is_best=True)

    status = "best saved" if improved else f"no improve {epochs_without_improvement}/{PATIENCE}"
    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} | "
        f"train_c={train_metrics['c_index']:.3f} valid_c={valid_metrics['c_index']:.3f} | "
        f"valid_HR={valid_metrics['hr_high_vs_low']:.2f} "
        f"(95% CI {valid_metrics['hr_ci_low']:.2f}-{valid_metrics['hr_ci_high']:.2f}) | "
        f"logrank_p={valid_metrics['logrank_p_value']:.3g} | "
        f"lr={current_lr:.2e} | best_epoch={best_epoch} ({status})"
    )

    if epoch % LOG_EVERY_EPOCHS == 0 or improved:
        plot_training_history(history, valid_metrics.get("prediction_df"))

    if epoch % SAVE_EVERY_EPOCHS == 0:
        save_checkpoint(M1_MODEL_DIR / f"checkpoint_epoch_{epoch:03d}.pt", epoch, checkpoint_metrics, is_best=False)

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best {MONITOR_METRIC}={best_score:.4f}")
        break

print("training finished")
print("best_epoch:", best_epoch)
print("best_score:", best_score)
print("best checkpoint:", BEST_CHECKPOINT_PATH)
