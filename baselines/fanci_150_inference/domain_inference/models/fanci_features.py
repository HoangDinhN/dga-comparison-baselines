from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

# Constants, order, and edge cases match FANCI commit d6c7d0885f95a6ef1a6e0efce66e50e22b47e0a2.
HEX_DIGITS = set("0123456789abcdef")
VOWELS = set("aeiou")
PARTS_MAX_CONSIDERED = 4
IPV4_PATTERN = re.compile(
    r"(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}"
    r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])"
)

NGRAM_STATS = ("std", "median", "mean", "min", "max", "q25", "q75")
FEATURE_NAMES = [
    "domain_length",
    "parts_1", "parts_2", "parts_3", "parts_4plus",
    "vowel_ratio", "digit_ratio", "contains_ipv4", "contains_digit",
    "has_valid_tld", "contains_one_char_subdomain", "contains_www_dot",
    "mean_suffix_free_part_length", "prefix_repetition", "character_diversity",
    "contains_only_digits_part", "contains_tld_as_infix",
]
for _n in (1, 2, 3):
    FEATURE_NAMES.extend([f"ngram{_n}_{name}" for name in NGRAM_STATS])
FEATURE_NAMES.extend([
    "hex_part_ratio", "underscore_ratio", "alphabet_size", "shannon_entropy",
    "repeated_char_ratio", "consecutive_consonant_ratio", "consecutive_digit_ratio",
])
assert len(FEATURE_NAMES) == 45


def normalize_domain(domain: object) -> str:
    """Apply the exact normalization used by the training notebook."""
    return str(domain).strip().lower().rstrip(".")


def load_public_suffixes(path: str | Path) -> tuple[frozenset[str], frozenset[str]]:
    """Parse FANCI's pinned public_suffix.txt representation."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries = [
        "." + line.strip().lower()
        for line in lines
        if line.strip() and not line.startswith(("/", "*"))
    ]
    valid_tlds = frozenset(item for item in entries if len(item.split(".")) == 2)
    return valid_tlds, frozenset(entries)


class OfficialFANCI45Extractor:
    """The 45-dimensional feature extractor from the official FANCI source."""

    def __init__(self, valid_tlds: Iterable[str], valid_public_suffixes: Iterable[str]):
        self.valid_tlds = frozenset(str(item) for item in valid_tlds)
        self.valid_public_suffixes = frozenset(str(item) for item in valid_public_suffixes)
        if not self.valid_tlds or not self.valid_public_suffixes:
            raise ValueError("FANCI Public Suffix List data must not be empty")

    def _remove_public_suffix(self, parts: Sequence[str]) -> tuple[tuple[str, ...], str]:
        match = ""
        if len(parts) < 2:
            return tuple(parts), match
        for index in range(len(parts)):
            match = "." + ".".join(parts[index:])
            if match in self.valid_public_suffixes:
                return tuple(parts[:index]), match
        # This surprising no-match behavior is present in the official source.
        return tuple(parts), match

    @staticmethod
    def _stats(values: Sequence[int]) -> list[float]:
        if not values:
            return [-1.0] * 7
        array = np.asarray(values, dtype=np.float64)
        return [
            float(array.std()), float(np.median(array)), float(array.mean()),
            float(array.min()), float(array.max()),
            float(np.percentile(array, 25)), float(np.percentile(array, 75)),
        ]

    @staticmethod
    def _run_ratio(parts: Sequence[str], predicate, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        total = 0
        for part in parts:
            run = 0
            for char in part:
                if predicate(char):
                    run += 1
                else:
                    if run > 1:
                        total += run
                    run = 0
            if run > 1:
                total += run
        return total / denominator

    def extract(self, domain: object) -> np.ndarray:
        domain = str(domain).strip().lower()
        dot_split = tuple(domain.split("."))
        suffix_free_parts, public_suffix = self._remove_public_suffix(dot_split)
        suffix_free = "".join(suffix_free_parts)
        suffix_free_length = len(suffix_free)

        vector: list[float] = [float(len(domain))]

        part_feature = [0.0] * PARTS_MAX_CONSIDERED
        split_length = len(suffix_free_parts)
        if split_length >= PARTS_MAX_CONSIDERED:
            part_feature[-1] = 1.0
        elif split_length > 0:
            part_feature[split_length - 1] = 1.0
        vector.extend(part_feature)

        alpha_count = sum(char.isalpha() for char in suffix_free)
        vowel_count = sum(char in VOWELS for char in suffix_free)
        digit_count = sum(char.isdigit() for char in suffix_free)
        vector.append(vowel_count / alpha_count if alpha_count else 0.0)
        vector.append(digit_count / suffix_free_length if suffix_free_length else 0.0)
        vector.append(float(bool(IPV4_PATTERN.search(domain))))
        vector.append(float(any(char.isdigit() for char in domain)))
        vector.append(float(bool(public_suffix)))

        one_char_parts = dot_split[:-1] if len(dot_split) > 2 else dot_split
        vector.append(float(any(len(part) == 1 for part in one_char_parts)))
        vector.append(float("www." in domain))
        vector.append(
            sum(map(len, suffix_free_parts)) / len(suffix_free_parts)
            if suffix_free_parts else 0.0
        )
        vector.append(float((domain + domain).find(domain, 1, -1) != -1))

        unigram_counter = Counter(suffix_free)
        vector.append(len(unigram_counter) / suffix_free_length if suffix_free_length else 0.0)

        # Official code tests "no alphabetic characters", despite the feature name.
        vector.append(float(any(not any(char.isalpha() for char in part) for part in dot_split)))
        vector.append(float(any(("." + part) in self.valid_tlds for part in suffix_free_parts)))

        unigram_counts: list[int] | None = None
        for ngram_size in (1, 2, 3):
            counts = Counter(
                suffix_free[index:index + ngram_size]
                for index in range(max(0, suffix_free_length - ngram_size + 1))
            )
            values = list(counts.values())
            if ngram_size == 1:
                unigram_counts = values
            vector.extend(self._stats(values))

        vector.append(
            sum(all(char in HEX_DIGITS for char in part) for part in suffix_free_parts)
            / len(suffix_free_parts)
            if suffix_free_parts else 0.0
        )
        vector.append(suffix_free.count("_") / suffix_free_length if suffix_free_length else 0.0)
        vector.append(float(len(unigram_counts or [])))

        if unigram_counts:
            counts_array = np.asarray(unigram_counts, dtype=np.float64)
            probabilities = counts_array / counts_array.sum()
            entropy = float(-(probabilities * np.log2(probabilities)).sum())
            repeated = float(np.sum(counts_array > 1) / len(counts_array))
        else:
            entropy = 0.0
            repeated = 0.0
        vector.extend([entropy, repeated])

        vector.append(self._run_ratio(
            suffix_free_parts,
            lambda char: char.isalpha() and char not in VOWELS,
            suffix_free_length,
        ))
        vector.append(self._run_ratio(
            suffix_free_parts, lambda char: char.isdigit(), suffix_free_length
        ))

        result = np.asarray(vector, dtype=np.float32)
        if result.shape != (45,) or not np.isfinite(result).all():
            raise ValueError(f"Invalid FANCI feature vector for domain={domain!r}: shape={result.shape}")
        return result

    def extract_many(self, domains: Sequence[object], n_jobs: int = 1) -> np.ndarray:
        if not domains:
            return np.empty((0, 45), dtype=np.float32)
        if n_jobs == 1:
            rows = [self.extract(domain) for domain in domains]
        else:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                delayed(self.extract)(domain) for domain in domains
            )
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape != (len(domains), 45) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid FANCI feature matrix shape={matrix.shape}")
        return matrix


def extract_features_batch(
    domains: Sequence[object],
    extractor: OfficialFANCI45Extractor,
    n_jobs: int = 1,
) -> np.ndarray:
    return extractor.extract_many(domains, n_jobs=n_jobs)
