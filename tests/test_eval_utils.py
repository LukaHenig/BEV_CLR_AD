import pytest

torch = pytest.importorskip("torch")
eval_module = pytest.importorskip("eval")


def test_update_metrics_accumulates_and_computes_iou():
    metrics = {"example_intersections": torch.tensor(2.0), "example_unions": torch.tensor(4.0), "example_iou": 0.0}
    metrics_single = {
        "example_intersections": torch.tensor(3.0),
        "example_unions": torch.tensor(5.0),
        "example_iou": torch.tensor(0.0),
    }

    eval_module.update_metrics("example", metrics, metrics_single)

    assert torch.equal(metrics["example_intersections"], torch.tensor(5.0))
    assert torch.equal(metrics["example_unions"], torch.tensor(9.0))
    expected_iou = 100 * 5.0 / (9.0 + 1e-4)
    assert torch.allclose(metrics["example_iou"], torch.tensor(expected_iou))


def test_update_range_metrics_iterates_over_ranges():
    base_metrics = {
        "obj_0_20_intersections": torch.tensor(0.0),
        "obj_0_20_unions": torch.tensor(0.0),
        "obj_0_20_iou": torch.tensor(0.0),
        "obj_20_35_intersections": torch.tensor(0.0),
        "obj_20_35_unions": torch.tensor(0.0),
        "obj_20_35_iou": torch.tensor(0.0),
        "obj_35_50_intersections": torch.tensor(0.0),
        "obj_35_50_unions": torch.tensor(0.0),
        "obj_35_50_iou": torch.tensor(0.0),
    }

    metrics_single = {
        "obj_0_20_intersections": torch.tensor(1.0),
        "obj_0_20_unions": torch.tensor(2.0),
        "obj_20_35_intersections": torch.tensor(3.0),
        "obj_20_35_unions": torch.tensor(4.0),
        "obj_35_50_intersections": torch.tensor(5.0),
        "obj_35_50_unions": torch.tensor(6.0),
    }

    eval_module.update_range_metrics("obj", base_metrics, metrics_single)

    assert torch.equal(base_metrics["obj_0_20_intersections"], torch.tensor(1.0))
    assert torch.equal(base_metrics["obj_20_35_unions"], torch.tensor(4.0))
    expected_iou = 100 * 5.0 / (6.0 + 1e-4)
    assert torch.allclose(base_metrics["obj_35_50_iou"], torch.tensor(expected_iou))


def test_calculate_best_map_ious_and_thresholds_selects_maxima():
    intersections = torch.tensor([[0.2, 0.5, 0.7]])
    unions = torch.tensor([[1.0, 1.0, 1.0]])
    thresholds = torch.tensor([0.1, 0.2, 0.3])

    best_map_ious, best_thresholds, best_map_mean_iou = eval_module.calculate_best_map_ious_and_thresholds(
        intersections=intersections, unions=unions, thresholds=thresholds
    )

    assert torch.allclose(best_map_ious, torch.tensor([0.7]))
    assert torch.allclose(best_thresholds, torch.tensor([0.3]))
    assert torch.allclose(best_map_mean_iou, torch.tensor(0.7))
