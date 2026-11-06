#!/usr/bin/env python3
"""Exact round elimination for ORDERED configuration vectors (Python 3.9+).

Use this script when each coordinate has a fixed port/input role: ["a", "b"]
and ["b", "a"] are distinct configurations. Both sides retain their own local
coordinate order, independently of the other endpoint. For C5, increasing
black-local port numbers can assign the five game roles before play. Neither
operator permutes coordinates or reassigns roles. Use round_eliminator.py for
ordinary unordered multiset constraints. The two arities may differ.

Usage:
  python round_eliminator_vectors.py problem.json -o RE_vectors_out.json
  python round_eliminator_vectors.py problem.json --operator Rbar -o Rbar_vectors.json
  python round_eliminator_vectors.py problem.json --two-step -o RE_vectors_two_step.json
  python round_eliminator_vectors.py RE_vectors_out.json --operator Rbar -o next.json
  python round_eliminator_vectors.py -o - < problem.json > result.json

Input/output JSON schema (explicitly list every permitted ordered vector):
  {"alphabet": ["a", "b"],
   "node_constraint": [["a", "b"]],
   "edge_constraint": [["a", "a"], ["b", "b"]]}
node_constraint is the white/player side; edge_constraint is the black/game
side. Aliases: labels for alphabet, node/white for node_constraint, and
edge/black for edge_constraint. Additional metadata is ignored, not copied.
Each side has one positive arity. For an empty constraint [], provide its
top-level node_arity or edge_arity; these fields are emitted when needed.
An empty alphabet is allowed when both constraints are empty with arities.

R (default) takes maximal universal boxes of edge_constraint and existentially
lifts node_constraint; Rbar swaps the sides. --two-step applies Rbar(R(P)).
--operator and --two-step are mutually exclusive. New labels are nonempty
subsets of previous labels, encoded as JSON arrays, nested on further steps.
Subset member order and duplicate subset members are ignored; configuration
order and repetitions are preserved. Output uses the same problem schema and
can be fed back into this script. Only maximal universal configurations are
retained. A subset-array entry is one label, not an additional wildcard rule.

Output defaults to RE_vectors_out.json. Omit input or use - to read stdin;
-o - writes JSON to stdout. Exit status: 0 success, 2 invalid input/options or
I/O error. Uses only the Python standard library. Exact enumeration can be
expensive. The universal computation uses coordinatewise union/intersection
closure with a maximal antichain and no permutation matching.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import itertools
import json
import sys
from typing import Any, Iterable

Label = frozenset[str]  # Canonical JSON encodings of previous-round labels.
Box = tuple[Label, ...]  # Fixed coordinate order, never sorted.


def _label_key(label: Any) -> str:
    if isinstance(label, str):
        return json.dumps(label, ensure_ascii=False)
    if not isinstance(label, list) or not label:
        raise ValueError("labels must be strings or nonempty nested subset arrays")
    members = sorted({_label_key(member) for member in label})
    return "[" + ",".join(members) + "]"


def _parse_problem(raw: dict[str, Any]):
    if not isinstance(raw, dict):
        raise ValueError("problem must be a JSON object")
    alphabet = raw.get("alphabet", raw.get("labels"))
    if not isinstance(alphabet, list):
        raise ValueError("alphabet must be an array of labels")
    encoded = [_label_key(label) for label in alphabet]
    if len(set(encoded)) != len(encoded):
        raise ValueError("alphabet labels must be unique")
    allowed = set(encoded)

    def constraint(name: str, aliases: tuple[str, ...]):
        value = next((raw[key] for key in (name, *aliases) if key in raw), None)
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array of configuration vectors")
        vectors = set()
        for conf in value:
            if not isinstance(conf, list) or not conf:
                raise ValueError(f"every {name} vector must be a nonempty array")
            vector = tuple(_label_key(label) for label in conf)
            if any(label not in allowed for label in vector):
                raise ValueError(f"every {name} entry must be in alphabet")
            vectors.add(vector)
        arities = {len(conf) for conf in vectors}
        if len(arities) > 1:
            raise ValueError(f"all {name} vectors must have equal size")
        arity_key = "node_arity" if name == "node_constraint" else "edge_arity"
        arity = raw.get(arity_key, next(iter(arities), None))
        if type(arity) is not int or arity < 1:
            raise ValueError(f"{arity_key} must be a positive integer for an empty constraint")
        if arities and arities != {arity}:
            raise ValueError(f"{arity_key} does not match {name} vector length")
        return vectors, arity

    nodes, node_arity = constraint("node_constraint", ("node", "white"))
    edges, edge_arity = constraint("edge_constraint", ("edge", "black"))
    return nodes, edges, node_arity, edge_arity


def _box_key(box: Box):
    return tuple(tuple(sorted(part)) for part in box)


def _dominates(upper: Box, lower: Box) -> bool:
    return len(upper) == len(lower) and all(a >= b for a, b in zip(upper, lower))


def _newre(configurations: Iterable[tuple[str, ...]]) -> set[Box]:
    """Maximal Cartesian boxes contained in the ordered input relation.

    Combine matching coordinates with one union and the other intersections.
    Maintain a maximal antichain; queue new boxes until closure. In particular,
    there is no permutation loop or multiset canonicalization.
    """
    current: set[Box] = set()
    pending: deque[Box] = deque()

    def insert(box: Box):
        if not all(box) or any(_dominates(old, box) for old in current):
            return
        current.difference_update([old for old in current if _dominates(box, old)])
        current.add(box)
        pending.append(box)

    for conf in sorted(configurations):
        insert(tuple(frozenset((label,)) for label in conf))
    while pending:
        left = pending.popleft()
        if left not in current:
            continue
        for right in sorted(current, key=_box_key):
            intersections = tuple(a & b for a, b in zip(left, right))
            empty = [i for i, part in enumerate(intersections) if not part]
            if len(empty) > 1:
                continue
            for u in (empty if empty else range(len(left))):
                union = left[u] | right[u]
                if union in (left[u], right[u]):
                    continue  # The result is already dominated by an operand.
                parts = list(intersections)
                parts[u] = union
                insert(tuple(parts))
    return current


def _existential_constraint(candidates: set[Label], valid: set[tuple[str, ...]]) -> set[Box]:
    # A vector of subset labels is legal iff it contains some input vector
    # coordinate by coordinate. Generate only choices with such a witness.
    containing = {
        label: sorted((subset for subset in candidates if label in subset), key=lambda s: tuple(sorted(s)))
        for conf in valid for label in conf
    }
    return {tuple(box) for conf in valid
            for box in itertools.product(*(containing[label] for label in conf))}


def _serialize_box(box: Box) -> list[list[Any]]:
    return [[json.loads(label) for label in sorted(part)] for part in box]


def round_eliminate(problem: dict[str, Any], operator: str = "R") -> dict[str, Any]:
    """Apply R or Rbar with both constraints interpreted as ordered vectors."""
    if operator not in {"R", "Rbar"}:
        raise ValueError("operator must be 'R' or 'Rbar'")
    nodes, edges, node_arity, edge_arity = _parse_problem(problem)
    universal = _newre(edges if operator == "R" else nodes)
    candidates = {part for box in universal for part in box}
    existential = _existential_constraint(candidates, nodes if operator == "R" else edges)
    new_nodes, new_edges = (existential, universal) if operator == "R" else (universal, existential)
    output = {
        "alphabet": [[json.loads(label) for label in sorted(subset)]
                     for subset in sorted(candidates, key=lambda s: (len(s), tuple(sorted(s))))],
        "node_constraint": [_serialize_box(box) for box in sorted(new_nodes, key=_box_key)],
        "edge_constraint": [_serialize_box(box) for box in sorted(new_edges, key=_box_key)],
    }
    # Preserve arity when an empty relation has no vectors from which to infer
    # it. This also makes unsatisfiable results valid inputs to another step.
    if not new_nodes:
        output["node_arity"] = node_arity
    if not new_edges:
        output["edge_arity"] = edge_arity
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="problem JSON file (default: stdin)")
    parser.add_argument("--output", "-o", default="RE_vectors_out.json",
                        help="JSON output path (default: RE_vectors_out.json)")
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--operator", choices=("R", "Rbar"), default="R")
    operations.add_argument("--two-step", action="store_true", help="apply Rbar(R(problem))")
    args = parser.parse_args()
    try:
        input_context = (open(args.input, encoding="utf-8") if args.input not in (None, "-")
                         else nullcontext(sys.stdin))
        with input_context as stream:
            problem = json.load(stream)
        output = round_eliminate(problem, "R" if args.two_step else args.operator)
        if args.two_step:
            output = round_eliminate(output, "Rbar")
        output_context = (open(args.output, "w", encoding="utf-8") if args.output != "-"
                          else nullcontext(sys.stdout))
        with output_context as stream:
            json.dump(output, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"round_eliminator_vectors: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
