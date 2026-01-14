import json
import sys

import yaml


def validate_model():
    """Checks model accuracy against a threshold."""

    # Загрузка параметров
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    accuracy_min = params.get("accuracy_min", 0.0)

    # Загрузка метрик
    with open("metrics/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    accuracy = metrics.get("accuracy", 0.0)

    # Проверка
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {accuracy_min:.4f}")

    if accuracy < accuracy_min:
        print("Model validation FAILED: Accuracy is below the threshold.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Model validation PASSED.")


if __name__ == "__main__":
    validate_model()
