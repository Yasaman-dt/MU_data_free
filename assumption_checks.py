import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- head finder (same spirit as your helper) ----------
def _last_linear(module: nn.Module) -> nn.Linear:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            return m
    raise AttributeError("No nn.Linear found inside the given module.")

def get_head_and_dims(net: nn.Module):
    """
    Returns (head_module, final_linear, D, C)
    Works for ResNet (.fc), ViT (.head/.heads), Swin (.head), etc.
    """
    for attr in ("heads", "head", "classifier", "fc", "classif"):
        if hasattr(net, attr):
            head = getattr(net, attr)
            if isinstance(head, nn.Linear):
                return head, head, head.in_features, head.out_features
            last = _last_linear(head)
            return head, last, last.in_features, last.out_features

    if hasattr(net, "get_classifier"):
        head = net.get_classifier()
        if isinstance(head, nn.Linear):
            return head, head, head.in_features, head.out_features
        last = _last_linear(head)
        return head, last, last.in_features, last.out_features

    raise AttributeError("Could not locate classifier layer.")

# --------- helpers ----------
def _as_forget_set(class_to_remove):
    if class_to_remove is None:
        return set()
    if isinstance(class_to_remove, (list, tuple, set)):
        return set(int(x) for x in class_to_remove)
    return {int(class_to_remove)}

def _onehot(labels, C):
    y = torch.zeros(labels.shape[0], C, device=labels.device, dtype=torch.float32)
    return y.scatter_(1, labels.view(-1, 1), 1.0)

def _s_from_ce(logits, targets_used, sign=+1.0):
    # per-sample grad wrt logits for CE is (p - onehot)
    p = F.softmax(logits, dim=1)
    s = p - _onehot(targets_used, logits.shape[1])
    return s * float(sign)

def _targets_used(method, logits, true_labels, forget_set, seed=0):
    """
    Returns the targets actually used in the method's forget loss (CE-based methods).
    """
    C = logits.shape[1]
    method = method.upper()

    if method in {"NG", "NGFT", "NGFTW"}:
        return true_labels

    if method == "RL":
        # RandomLabels: choose random labels from classes NOT in forget_set
        allowed = [k for k in range(C) if k not in forget_set]
        if len(allowed) == 0:
            raise ValueError("RL: allowed label set is empty (forget_set covers all classes).")
        allowed = torch.tensor(allowed, device=logits.device, dtype=torch.long)

        g = torch.Generator(device=logits.device)
        g.manual_seed(int(seed))
        idx = torch.randint(low=0, high=allowed.numel(), size=true_labels.shape, generator=g, device=logits.device)
        return allowed[idx]

    if method == "BS":
        # BoundaryShrink: "nearest but incorrect" (your top-2 logic)
        top2 = torch.topk(logits, k=2, dim=1).indices
        pred_label = torch.where(top2[:, 0] == true_labels, top2[:, 1], top2[:, 0])
        return pred_label.detach()

    if method == "FT":
        # FineTuning doesn't use forget loss; target not meaningful
        return None

    raise NotImplementedError(
        f"Targets for method={method} not implemented in this checker. "
        "If this method uses CE/KL, we can add it."
    )

def _ce_sign(method):
    method = method.upper()
    if method in {"NG", "NGFT", "NGFTW"}:
        return -1.0  # -CE
    if method in {"RL", "BS"}:
        return +1.0  # +CE (with different targets)
    if method == "FT":
        return None
    raise NotImplementedError(f"Sign for method={method} not implemented.")

def _get_pf_tensors(forget_loader):
    ds = forget_loader.dataset
    if not isinstance(ds, torch.utils.data.TensorDataset):
        raise TypeError("Expected TensorDataset for forget_loader.dataset.")
    X, y = ds.tensors
    return X, y


@torch.no_grad()
def run_assumption_checks(
    net,
    forget_loader,
    class_to_remove,
    method,
    device="cuda",
    N=50000,
    seed=0,
    eps=1e-12,
    print_A=True,
    teacher_model=None,      # <- NEW
    be_shadow_bias=5.0,      # <- NEW (for BE approximation)
):
    """
    Checks:
      - (4)-style sign pattern on s for samples from p_f
      - (5)(6) alignment signs using cheap estimator:
          mu = E[g(z')]
          v_k = E[s_k(z) g(z)]
          A_k = v_k^T mu

    Uses your empirical p_f from forget_loader.dataset.
    """

    forget_set = _as_forget_set(class_to_remove)

    head, _, D, C = get_head_and_dims(net)
    head = head.to(device).eval()

    X_all, y_all = _get_pf_tensors(forget_loader)
    X_all = X_all.to(device)
    y_all = y_all.to(device)

    # make sure we only use samples whose true label is in forget_set
    mask = torch.zeros_like(y_all, dtype=torch.bool)
    for k in forget_set:
        mask |= (y_all == k)
    Xf = X_all[mask]
    yf = y_all[mask]

    if Xf.shape[0] == 0:
        raise ValueError("No samples in forget_loader matching class_to_remove.")

    N_use = min(int(N), Xf.shape[0])

    # two independent subsamples (to mimic z and z' independent)
    g1 = torch.Generator(device=device); g1.manual_seed(int(seed))
    g2 = torch.Generator(device=device); g2.manual_seed(int(seed) + 999)

    idx1 = torch.randperm(Xf.shape[0], generator=g1, device=device)[:N_use]
    idx2 = torch.randperm(Xf.shape[0], generator=g2, device=device)[:N_use]

    X1, y1 = Xf[idx1], yf[idx1]
    X2, y2 = Xf[idx2], yf[idx2]

    logits1 = head(X1)
    methodU = method.upper()

    sign = None  # <-- IMPORTANT: define sign for all methods

    if methodU in {"NG","NGFT","NGFTW","RL","BS"}:
        sign = _ce_sign(methodU)
        targets_used = _targets_used(methodU, logits1, y1, forget_set, seed=seed)
        if targets_used is None:
            print(f"[Assumption checks] method={method}: no forget-loss targets. Skipping.")
            return None
        s1 = _s_from_ce(logits1, targets_used, sign=sign)

    elif methodU == "DELETE":
        if teacher_model is None:
            raise ValueError("DELETE check needs teacher_model")
        teacher_model = teacher_model.to(device).eval()
        head_t, _, _, _ = get_head_and_dims(teacher_model)

        t_logits = head_t(X1).clone()
        t_logits.scatter_(1, y1.view(-1, 1), -1e9)
        q = F.softmax(t_logits, dim=1)
        p = F.softmax(logits1, dim=1)
        s1 = p - q

    elif methodU == "SCRUB":
        # In your SCRUB code, forget loss is: loss_kd_forget = -KD(student, teacher)
        # KD is KL(q || p) with temperature T, so grad ~ -(p - q) = (q - p) (up to +T scale)
        if teacher_model is None:
            raise ValueError("SCRUB check needs teacher_model (pass a frozen copy of the original net).")

        teacher_model = teacher_model.to(device).eval()
        head_t, _, _, _ = get_head_and_dims(teacher_model)

        t_logits = head_t(X1)
        q = F.softmax(t_logits, dim=1)
        p = F.softmax(logits1, dim=1)

        # grad scale is positive (≈ T), sign is what matters here:
        s1 = (q - p)


    elif methodU == "BE":
        shadow_logit = torch.full((logits1.size(0), 1), float(be_shadow_bias), device=device)
        logits_aug = torch.cat([logits1, shadow_logit], dim=1)
        p_aug = F.softmax(logits_aug, dim=1)
        s1 = p_aug[:, :logits1.size(1)]

    elif methodU == "SCAR":
        print("[SCAR] not applicable for logit-gradient assumptions.")
        return None

    else:
        raise NotImplementedError(...)

    sign_str = f"{sign:+.0f}" if sign is not None else "N/A"
    print(f"method={method}  CE-sign={sign_str}  N={N_use}")
    
    # ------------------- Assumption (4)-style check -------------------
    # For each sample (true forget label y):
    #   require s_y > 0
    #   and for retain classes r (not in forget_set): s_r < 0
    retain_classes = [k for k in range(C) if k not in forget_set]
    rows = torch.arange(N_use, device=device)

    s_true = s1[rows, y1]  # component for the TRUE forget label
    ok_true = (s_true > eps)

    if len(retain_classes) > 0:
        s_ret = s1[:, retain_classes]
        ok_ret = (s_ret < -eps).all(dim=1)
    else:
        ok_ret = torch.ones_like(ok_true, dtype=torch.bool)

    ok_both = ok_true & ok_ret

    print("\n=== Assumption (3)-style sign check on p_f ===")
    print(f"method={method}  CE-sign={sign_str}  N={N_use}")
    print(f"Fraction s_trueLabel>0:         {ok_true.float().mean().item():.6f}")
    print(f"Fraction all retain comps <0:  {ok_ret.float().mean().item():.6f}")
    print(f"Fraction both:                 {ok_both.float().mean().item():.6f}")
    print(f"min s_trueLabel: {s_true.min().item():+.6e}")
    if len(retain_classes) > 0:
        print(f"max s_retain:    {s1[:, retain_classes].max().item():+.6e}  (should be < 0)")

    # ------------------- Assumptions (5) cheap estimator -------------------
    mu = X2.float().mean(dim=0)  # [D]
    v = (s1.float().unsqueeze(-1) * X1.float().unsqueeze(1)).mean(dim=0)  # [C, D]
    A = v @ mu  # [C]

    print("\n=== Cheap estimator for (5)(6): A_k = v_k^T mu ===")
    # check signs: forget classes positive, retain negative
    ok_forget = True
    for k in sorted(list(forget_set)):
        ok_forget &= (A[k].item() > 0)
    ok_retain = True
    for k in retain_classes:
        ok_retain &= (A[k].item() < 0)

    print(f"A_forget > 0 for all forget classes?   {ok_forget}")
    print(f"A_retain < 0 for all retain classes?   {ok_retain}")

    if print_A:
        for k in range(C):
            tag = "FORGET" if k in forget_set else "retain"
            print(f"k={k:3d}  A_k={A[k].item():+.6e}  ({tag})")

    return {
        "N": N_use,
        "method": method,
        "frac_ok_true": ok_true.float().mean().item(),
        "frac_ok_retain": ok_ret.float().mean().item(),
        "frac_ok_both": ok_both.float().mean().item(),
        "A": A.detach().cpu(),
    }