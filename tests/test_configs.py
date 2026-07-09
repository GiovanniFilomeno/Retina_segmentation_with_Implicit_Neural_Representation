from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("filename", "task", "num_classes"),
    [
        ("fives_binary.yaml", "binary", None),
        ("ravir_multiclass.yaml", "multiclass", 3),
    ],
)
def test_research_config_is_parseable_and_explicit(filename, task, num_classes):
    config_path = Path(__file__).parents[1] / "configs" / filename

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["schema_version"] == 1
    assert config["experiment"]["task"] == task
    assert config["experiment"]["status"] == "illustrative_unvalidated_configuration"
    assert config["experiment"]["seed"] == 42
    assert config["model"]["task"] == task
    assert config["model"]["num_classes"] == num_classes
    assert config["evaluation"]["report_per_image"] is True
