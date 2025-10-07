import math

import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")


def test_requires_grad_toggles_parameters():
    model = torch.nn.Linear(4, 2)
    params = list(model.parameters())

    train.requires_grad(params, False)
    assert all(not p.requires_grad for p in params)

    train.requires_grad(params, True)
    assert all(p.requires_grad for p in params)


def test_fetch_optimizer_and_scheduler_step():
    model = torch.nn.Linear(3, 1)
    optimizer, scheduler = train.fetch_optimizer(
        lr=1e-3, wdecay=1e-2, epsilon=1e-8, num_steps=5, params=model.parameters()
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    stepped_lr = optimizer.param_groups[0]["lr"]

    assert not math.isclose(initial_lr, 0.0)
    assert stepped_lr != initial_lr


def test_simple_loss_reduces_on_better_predictions():
    criterion = train.SimpleLoss(pos_weight=1.0)
    y_target = torch.ones((2, 3))
    valid_mask = torch.ones_like(y_target)

    poor_logits = torch.zeros_like(y_target)
    better_logits = torch.ones_like(y_target) * 5.0

    poor_loss = criterion(poor_logits, y_target, valid_mask)
    better_loss = criterion(better_logits, y_target, valid_mask)

    assert better_loss < poor_loss


def test_sigmoid_focal_loss_balances_easy_and_hard_examples():
    criterion = train.SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    easy_logits = torch.tensor([[6.0, -6.0], [-6.0, 6.0]])
    hard_logits = torch.zeros_like(easy_logits)

    easy_loss = criterion(easy_logits, targets)
    hard_loss = criterion(hard_logits, targets)

    assert hard_loss > easy_loss


def test_grad_acc_metrics_accumulates_and_normalizes():
    metrics_single_pass = {
        "loss": torch.tensor(2.0),
        "accuracy": torch.tensor(0.5),
        "map_seg_thresholds": torch.arange(3.0),
    }
    metrics_mean_grad_acc = {"loss": 0.0, "accuracy": 0.0, "map_seg_thresholds": None}

    result = train.grad_acc_metrics(metrics_single_pass, metrics_mean_grad_acc, internal_step=1, grad_acc=2)

    assert math.isclose(float(result["loss"]), 1.0)
    assert math.isclose(float(result["accuracy"]), 0.25)
    assert torch.equal(result["map_seg_thresholds"], metrics_single_pass["map_seg_thresholds"])
