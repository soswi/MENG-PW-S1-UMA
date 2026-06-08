import argparse
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from aq_experiments import (
    AQClassifier,
    evaluate_predictions,
    load_dataset,
    stratified_split,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VARIANTS = ("classic", "validation")
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "macro_f1")


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.pstdev(values)


def fmt(values, precision=4):
    mean, std = mean_std(values)
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def collect(values, variant, key):
    return [record[key] for record in values if record["variant"] == variant]


def run_dataset(path, seeds, beam_width, max_rules_per_class, train_ratio, validation_ratio):
    x_data, y_data, feature_names = load_dataset(path)
    records = []
    split_sizes = (0, 0, 0)
    cap_hits = Counter()

    for seed in seeds:
        x_train, y_train, x_val, y_val, x_test, y_test = stratified_split(
            x_data,
            y_data,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            random_state=seed,
        )
        split_sizes = (len(x_train), len(x_val), len(x_test))

        for variant in VARIANTS:
            model = AQClassifier(
                variant=variant,
                beam_width=beam_width,
                max_rules_per_class=max_rules_per_class,
                random_state=seed,
            )
            model.fit(x_train, y_train, feature_names, x_val, y_val)
            test_metrics = evaluate_predictions(y_test, model.predict(x_test))
            train_metrics = evaluate_predictions(y_train, model.predict(x_train))
            summary = model.rules_summary()

            per_class = Counter(rule.target_class for rule in model.rules)
            for label, count in per_class.items():
                if count >= max_rules_per_class:
                    cap_hits[f"{variant}:{label}"] += 1

            record = {
                "dataset": path.name,
                "seed": seed,
                "variant": variant,
                "rules": summary["rules"],
                "avg_conditions": summary["avg_conditions"],
                "total_conditions": summary["total_conditions"],
                "gap_accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
                "gap_macro_f1": train_metrics["macro_f1"] - test_metrics["macro_f1"],
            }
            for key in METRIC_KEYS:
                record[f"test_{key}"] = test_metrics[key]
                record[f"train_{key}"] = train_metrics[key]
            records.append(record)

    _print_dataset_report(
        path, x_data, y_data, feature_names, seeds, split_sizes, records, cap_hits,
        max_rules_per_class,
    )
    return records


def _print_dataset_report(
    path, x_data, y_data, feature_names, seeds, split_sizes, records, cap_hits,
    max_rules_per_class,
):
    print(f"\n=== {path.name} ===")
    print(
        f"rows={len(x_data)}, features={len(feature_names)}, "
        f"classes={dict(Counter(y_data))}"
    )
    print(
        f"split (per seed): train={split_sizes[0]}, "
        f"validation={split_sizes[1]}, test={split_sizes[2]}"
    )
    print(f"seeds={seeds}  (mean±std over {len(seeds)} seeds)")
    print("uwaga: precision/recall/f1 sa wazone wsparciem (weighted)")

    print("\n[jakosc na zbiorze TESTOWYM]")
    header = (
        f"{'variant':<11} {'accuracy':>15} {'precision':>15} {'recall':>15} "
        f"{'f1':>15} {'macro_f1':>15}"
    )
    print(header)
    for variant in VARIANTS:
        print(
            f"{variant:<11} "
            f"{fmt(collect(records, variant, 'test_accuracy')):>15} "
            f"{fmt(collect(records, variant, 'test_precision')):>15} "
            f"{fmt(collect(records, variant, 'test_recall')):>15} "
            f"{fmt(collect(records, variant, 'test_f1')):>15} "
            f"{fmt(collect(records, variant, 'test_macro_f1')):>15}"
        )

    print("\n[przeuczenie (luka train-test) oraz reguly]")
    header2 = (
        f"{'variant':<11} {'train_acc':>15} {'train_macroF1':>15} "
        f"{'gap_acc':>15} {'gap_macroF1':>15} {'rules':>13} {'avg_cond':>13}"
    )
    print(header2)
    for variant in VARIANTS:
        print(
            f"{variant:<11} "
            f"{fmt(collect(records, variant, 'train_accuracy')):>15} "
            f"{fmt(collect(records, variant, 'train_macro_f1')):>15} "
            f"{fmt(collect(records, variant, 'gap_accuracy')):>15} "
            f"{fmt(collect(records, variant, 'gap_macro_f1')):>15} "
            f"{fmt(collect(records, variant, 'rules'), 1):>13} "
            f"{fmt(collect(records, variant, 'avg_conditions'), 2):>13}"
        )

    print("\n[roznica parowana: validation - classic (ten sam podzial)]")
    for key, label in (("test_accuracy", "accuracy"), ("test_macro_f1", "macro_f1")):
        classic_vals = collect(records, "classic", key)
        validation_vals = collect(records, "validation", key)
        deltas = [v - c for c, v in zip(classic_vals, validation_vals)]
        wins = sum(1 for d in deltas if d > 1e-9)
        mean_delta = st.mean(deltas) if deltas else 0.0
        rounded = [round(d, 3) for d in deltas]
        print(
            f"  {label:<9} mean={mean_delta:+.4f}  "
            f"wins(val>classic)={wins}/{len(deltas)}  per-seed={rounded}"
        )

    if cap_hits:
        print(
            "\n  UWAGA: budzet regul (max_rules_per_class="
            f"{max_rules_per_class}) wyczerpany dla: {dict(cap_hits)}"
        )
        print(
            "  (limit moze maskowac roznice miedzy wariantami - rozwaz wyzsza "
            "wartosc --max-rules-per-class)"
        )


def write_csv(records, csv_path):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nZapisano wyniki per-seed do: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wieloziarnowe porownanie klasycznego AQ i AQ z ocena kompleksow "
            "na zbiorze walidacyjnym (z luka train-test)."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=[
            Path("car_evaluation.csv"),
            Path("mushrooms.csv"),
            Path("nursery.csv"),
        ],
        help="Sciezki do plikow CSV z danymi.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Lista ziaren losowych (kazde = osobny podzial train/val/test).",
    )
    parser.add_argument("--beam-width", type=int, default=30)
    parser.add_argument(
        "--max-rules-per-class",
        type=int,
        default=80,
        help=(
            "Limit regul na klase (jak w aq_experiments.py). Wyzsza wartosc, "
            "np. 200, zapobiega maskowaniu roznic przez limit."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Opcjonalna sciezka do zapisu wynikow per-seed w formacie CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_records = []
    for dataset_path in args.datasets:
        all_records.extend(
            run_dataset(
                path=dataset_path,
                seeds=args.seeds,
                beam_width=args.beam_width,
                max_rules_per_class=args.max_rules_per_class,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
        )
    if args.csv is not None:
        write_csv(all_records, args.csv)


if __name__ == "__main__":
    main()
