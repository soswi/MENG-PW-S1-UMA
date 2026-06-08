import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateStats:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    covered: int


@dataclass(frozen=True)
class Rule:
    target_class: str
    conditions: tuple
    score: float
    precision: float
    recall: float
    covered_positives: int

    @property
    def complexity(self):
        return len(self.conditions)


class AQClassifier:

    def __init__(
        self,
        variant,
        beam_width=30,
        max_rules_per_class=80,
        random_state=42,
    ):
        if variant not in {"classic", "validation"}:
            raise ValueError("variant must be 'classic' or 'validation'")
        self.variant = variant
        self.beam_width = beam_width
        self.max_rules_per_class = max_rules_per_class
        self.random_state = random_state
        self.rules = []
        self.default_class = None
        self.feature_names = []

    def fit(
        self,
        x_train,
        y_train,
        feature_names,
        x_validation=None,
        y_validation=None,
    ):
        self.feature_names = feature_names
        self.default_class = Counter(y_train).most_common(1)[0][0]
        self.rules = []

        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation or []
        self._y_validation = y_validation or []
        self._n_features = len(feature_names)
        self._rng = random.Random(self.random_state)
        self._score_cache = {}

        self._build_bit_masks()

        class_order = [label for label, _ in Counter(y_train).most_common()]
        for target_class in class_order:
            self._induce_rules_for_class(target_class)

        return self

    def predict(self, x_data):
        if self.default_class is None:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = []
        for row in x_data:
            matching_rules = [rule for rule in self.rules if self._rule_covers_row(rule, row)]
            if not matching_rules:
                predictions.append(self.default_class)
                continue

            best_rule = max(
                matching_rules,
                key=lambda rule: (
                    rule.score,
                    rule.precision,
                    rule.recall,
                    rule.covered_positives,
                    -rule.complexity,
                ),
            )
            predictions.append(best_rule.target_class)

        return predictions

    def rules_summary(self):
        total_complexity = sum(rule.complexity for rule in self.rules)
        return {
            "rules": float(len(self.rules)),
            "total_conditions": float(total_complexity),
            "avg_conditions": total_complexity / len(self.rules) if self.rules else 0.0,
        }

    def format_rules(self, limit=10):
        formatted = []
        for rule in self.rules[:limit]:
            if rule.conditions:
                parts = [
                    f"{self.feature_names[attr]}={value}"
                    for attr, value in sorted(rule.conditions)
                ]
                condition_text = " AND ".join(parts)
            else:
                condition_text = "TRUE"
            formatted.append(
                f"IF {condition_text} THEN class={rule.target_class} "
                f"(score={rule.score:.3f}, precision={rule.precision:.3f})"
            )
        return formatted

    def _induce_rules_for_class(self, target_class):
        positive_indices = {
            index for index, label in enumerate(self._y_train) if label == target_class
        }
        uncovered = set(positive_indices)
        rules_for_class = 0

        while uncovered and rules_for_class < self.max_rules_per_class:
            seed_index = self._choose_seed(uncovered)
            candidates = self._generate_star(seed_index, target_class)
            if not candidates:
                candidates = [self._full_complex_for_seed(seed_index)]

            best_complex = max(
                candidates,
                key=lambda complex_: self._candidate_sort_key(complex_, target_class),
            )

            covered_positive_indices = {
                index
                for index in uncovered
                if self._complex_covers_row(best_complex, self._x_train[index])
            }
            if not covered_positive_indices:
                best_complex = self._full_complex_for_seed(seed_index)
                covered_positive_indices = {seed_index}

            stats = self._score_candidate(best_complex, target_class)
            self.rules.append(
                Rule(
                    target_class=target_class,
                    conditions=tuple(sorted(best_complex)),
                    score=stats.f1,
                    precision=stats.precision,
                    recall=stats.recall,
                    covered_positives=stats.true_positives,
                )
            )

            uncovered -= covered_positive_indices
            rules_for_class += 1

    def _choose_seed(self, uncovered):
        ordered = sorted(uncovered)
        return ordered[self._rng.randrange(len(ordered))]

    def _generate_star(self, seed_index, target_class):
        seed = self._x_train[seed_index]
        negative_indices = [
            index for index, label in enumerate(self._y_train) if label != target_class
        ]

        candidates = {frozenset()}
        for negative_index in negative_indices:
            negative = self._x_train[negative_index]
            next_candidates = set()

            for complex_ in candidates:
                if not self._complex_covers_row(complex_, negative):
                    next_candidates.add(complex_)
                    continue

                used_attributes = {attribute for attribute, _ in complex_}
                specialized = False
                for attribute in range(self._n_features):
                    if attribute in used_attributes:
                        continue
                    if seed[attribute] == negative[attribute]:
                        continue
                    next_candidates.add(frozenset((*complex_, (attribute, seed[attribute]))))
                    specialized = True

                if not specialized:
                    next_candidates.add(complex_)

            candidates = set(self._prune_candidates(next_candidates, target_class))
            if not candidates:
                break

        return list(candidates)

    def _prune_candidates(self, candidates, target_class):
        unique_candidates = set(candidates)
        ordered = sorted(
            unique_candidates,
            key=lambda complex_: self._candidate_sort_key(complex_, target_class),
            reverse=True,
        )
        return ordered[: self.beam_width]

    def _candidate_sort_key(self, complex_, target_class):
        stats = self._score_candidate(complex_, target_class)
        return (
            stats.f1,
            stats.precision,
            stats.recall,
            stats.true_positives,
            -stats.false_positives,
            -len(complex_),
        )

    def _score_candidate(self, complex_, target_class):
        score_set = self._score_set_for_class(target_class)
        cache_key = (complex_, target_class, score_set)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if score_set == "validation":
            full_mask = self._full_validation_mask
            condition_masks = self._validation_condition_masks
            label_mask = self._validation_label_masks[target_class]
            total_positives = label_mask.bit_count()
        else:
            full_mask = self._full_train_mask
            condition_masks = self._train_condition_masks
            label_mask = self._train_label_masks[target_class]
            total_positives = label_mask.bit_count()

        covered_mask = full_mask
        for condition in complex_:
            covered_mask &= condition_masks.get(condition, 0)

        covered = covered_mask.bit_count()
        true_positives = (covered_mask & label_mask).bit_count()
        false_positives = covered - true_positives
        false_negatives = total_positives - true_positives

        precision = true_positives / covered if covered else 0.0
        recall = true_positives / total_positives if total_positives else 0.0
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 2 * true_positives / denominator if denominator else 0.0

        stats = CandidateStats(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            covered=covered,
        )
        self._score_cache[cache_key] = stats
        return stats

    def _score_set_for_class(self, target_class):
        if self.variant != "validation":
            return "train"
        if self._validation_label_masks[target_class].bit_count() == 0:
            return "train"
        return "validation"

    def _full_complex_for_seed(self, seed_index):
        seed = self._x_train[seed_index]
        return frozenset((attribute, seed[attribute]) for attribute in range(self._n_features))

    def _build_bit_masks(self):
        self._full_train_mask = (1 << len(self._x_train)) - 1
        self._full_validation_mask = (1 << len(self._x_validation)) - 1

        self._train_condition_masks = self._make_condition_masks(self._x_train)
        self._validation_condition_masks = self._make_condition_masks(self._x_validation)

        all_classes = set(self._y_train) | set(self._y_validation)
        self._train_label_masks = self._make_label_masks(self._y_train, all_classes)
        self._validation_label_masks = self._make_label_masks(
            self._y_validation, all_classes
        )

    def _make_condition_masks(self, x_data):
        masks = defaultdict(int)
        for row_index, row in enumerate(x_data):
            bit = 1 << row_index
            for attribute, value in enumerate(row):
                masks[(attribute, value)] |= bit
        return dict(masks)

    def _make_label_masks(self, y_data, all_classes):
        masks = {label: 0 for label in all_classes}
        for row_index, label in enumerate(y_data):
            masks[label] |= 1 << row_index
        return masks

    def _complex_covers_row(self, complex_, row):
        return all(row[attribute] == value for attribute, value in complex_)

    def _rule_covers_row(self, rule, row):
        return all(row[attribute] == value for attribute, value in rule.conditions)


def load_dataset(path):
    name = path.name.lower()
    rows = read_csv(path)

    if name == "car_evaluation.csv":
        feature_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        x_data = [tuple(row[:-1]) for row in rows]
        y_data = [row[-1] for row in rows]
        return x_data, y_data, feature_names

    header = rows[0]
    data_rows = rows[1:]
    if name == "mushrooms.csv":
        target_index = header.index("class")
    elif name == "nursery.csv":
        target_index = header.index("final evaluation")
    else:
        target_index = len(header) - 1

    feature_names = [
        column_name for index, column_name in enumerate(header) if index != target_index
    ]
    x_data = [
        tuple(value for index, value in enumerate(row) if index != target_index)
        for row in data_rows
    ]
    y_data = [row[target_index] for row in data_rows]
    return x_data, y_data, feature_names


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [row for row in csv.reader(csv_file) if row]


def stratified_split(
    x_data,
    y_data,
    train_ratio=0.6,
    validation_ratio=0.2,
    random_state=42,
):
    rng = random.Random(random_state)
    grouped_indices = defaultdict(list)
    for index, label in enumerate(y_data):
        grouped_indices[label].append(index)

    train_indices = []
    validation_indices = []
    test_indices = []

    for label in sorted(grouped_indices):
        indices = grouped_indices[label]
        rng.shuffle(indices)
        train_count, validation_count, test_count = split_counts(
            len(indices), train_ratio, validation_ratio
        )

        train_indices.extend(indices[:train_count])
        validation_indices.extend(indices[train_count : train_count + validation_count])
        test_indices.extend(
            indices[train_count + validation_count : train_count + validation_count + test_count]
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    return (
        [x_data[index] for index in train_indices],
        [y_data[index] for index in train_indices],
        [x_data[index] for index in validation_indices],
        [y_data[index] for index in validation_indices],
        [x_data[index] for index in test_indices],
        [y_data[index] for index in test_indices],
    )


def split_counts(count, train_ratio, validation_ratio):
    if count <= 0:
        return 0, 0, 0
    if count == 1:
        return 1, 0, 0
    if count == 2:
        return 1, 1, 0
    if count == 3:
        return 1, 1, 1
    if count == 4:
        return 2, 1, 1

    validation_count = max(1, round(count * validation_ratio))
    test_count = max(1, round(count * (1.0 - train_ratio - validation_ratio)))
    train_count = count - validation_count - test_count
    if train_count < 1:
        train_count = 1
        overflow = train_count + validation_count + test_count - count
        test_count = max(0, test_count - overflow)
    return train_count, validation_count, test_count


def evaluate_predictions(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    accuracy = correct / total if total else 0.0

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_f1 = 0.0

    for label in labels:
        true_positives = sum(
            1 for true, predicted in zip(y_true, y_pred) if true == label and predicted == label
        )
        false_positives = sum(
            1 for true, predicted in zip(y_true, y_pred) if true != label and predicted == label
        )
        false_negatives = sum(
            1 for true, predicted in zip(y_true, y_pred) if true == label and predicted != label
        )
        support = sum(1 for true in y_true if true == label)

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        weight = support / total if total else 0.0
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight
        macro_f1 += f1 / len(labels) if labels else 0.0

    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "macro_f1": macro_f1,
    }


def run_experiment(
    dataset_path,
    beam_width,
    max_rules_per_class,
    random_state,
    show_rules,
):
    x_data, y_data, feature_names = load_dataset(dataset_path)
    x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_split(
        x_data, y_data, random_state=random_state
    )

    print(f"\n=== {dataset_path.name} ===")
    print(
        f"rows={len(x_data)}, features={len(feature_names)}, "
        f"classes={dict(Counter(y_data))}"
    )
    print(
        f"split: train={len(x_train)}, validation={len(x_validation)}, test={len(x_test)}"
    )

    for variant in ("classic", "validation"):
        model = AQClassifier(
            variant=variant,
            beam_width=beam_width,
            max_rules_per_class=max_rules_per_class,
            random_state=random_state,
        )
        model.fit(x_train, y_train, feature_names, x_validation, y_validation)
        predictions = model.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        summary = model.rules_summary()

        print(f"\n{variant.upper()}")
        print(
            "accuracy={accuracy:.4f}, precision={precision:.4f}, "
            "recall={recall:.4f}, f1={f1:.4f}, macro_f1={macro_f1:.4f}".format(
                **metrics
            )
        )
        print(
            "rules={rules:.0f}, total_conditions={total_conditions:.0f}, "
            "avg_conditions={avg_conditions:.2f}".format(**summary)
        )

        if show_rules:
            for rule_text in model.format_rules(limit=show_rules):
                print(f"  {rule_text}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Porównanie klasycznego AQ oraz AQ z oceną kompleksów "
            "na zbiorze walidacyjnym."
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
        help="Ścieżki do plików CSV z danymi.",
    )
    parser.add_argument("--beam-width", type=int, default=30)
    parser.add_argument("--max-rules-per-class", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--show-rules",
        type=int,
        default=0,
        help="Ile pierwszych reguł wypisać dla każdego wariantu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset_path in args.datasets:
        run_experiment(
            dataset_path=dataset_path,
            beam_width=args.beam_width,
            max_rules_per_class=args.max_rules_per_class,
            random_state=args.random_state,
            show_rules=args.show_rules,
        )


if __name__ == "__main__":
    main()
