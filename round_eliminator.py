#!/usr/bin/env python3
"""Exact round elimination for UNORDERED multiset configurations (Python 3.9+).

Use this script for ordinary LCL constraints with interchangeable ports:
["a", "b"] and ["b", "a"] denote the same configuration, while repetitions
still count. Use round_eliminator_vectors.py when coordinate order matters.
The two constraint arities may differ, as in a biregular black-white graph.

Usage:
  python round_eliminator.py problem.json -o RE_out.json
  python round_eliminator.py problem.json --operator Rbar -o Rbar_out.json
  python round_eliminator.py problem.json --two-step -o RE_two_step.json
  python round_eliminator.py RE_out.json --operator Rbar -o next.json
  python round_eliminator.py -o - < problem.json > result.json

Input/output JSON schema (both constraints explicitly list allowed multisets):
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
multiplicities are preserved. Output uses the same problem schema and can be
fed back into this script. Only maximal universal configurations are retained.

Output defaults to RE_out.json. Omit input or use - to read stdin; -o - writes
JSON to stdout. Exit status: 0 success, 2 invalid input/options or I/O error.
Uses only the Python standard library. Exact enumeration can be expensive.
The universal computation uses NewRE union/intersection closure with arbitrary
permutation matching (Section 4, Towards Fully Automatic Distributed Lower
Bounds).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from contextlib import nullcontext
from typing import Any, Iterable

Label = frozenset[str]  # canonical JSON encodings of labels (supports subset labels)
Box = tuple[Label, ...]  # unordered multiset of per-position label sets


def _canon_box(parts: Iterable[Label]) -> Box:
    return tuple(sorted(parts, key=lambda s: (len(s), tuple(sorted(s)))))


def _match_inclusion(a: Box, b: Box) -> bool:
    """Whether a is dominated by b, allowing arbitrary port permutation."""
    if len(a) != len(b):
        return False
    order = sorted(range(len(a)), key=lambda i: len(a[i]))
    used: set[int] = set()

    def visit(k: int) -> bool:
        if k == len(order):
            return True
        x = a[order[k]]
        for j, y in enumerate(b):
            if j not in used and x <= y:
                used.add(j)
                if visit(k + 1):
                    return True
                used.remove(j)
        return False

    return visit(0)


def _maximal(boxes: Iterable[Box]) -> set[Box]:
    unique = set(boxes)
    return {b for b in unique if not any(b != c and _match_inclusion(b, c) for c in unique)}


def _newre(configurations: Iterable[tuple[str, ...]], arity: int) -> set[Box]:
    """Compute maximal universal boxes with the paper's NewRE closure."""
    current: set[Box] = {
        _canon_box(frozenset((label,)) for label in conf)
        for conf in configurations
    }
    if any(len(c) != arity for c in current):
        raise ValueError("constraint configurations have inconsistent arity")
    while True:
        generated = set(current)
        items = list(current)
        for left in items:
            for right in items:
                for perm in itertools.permutations(range(arity)):
                    for union_at in range(arity):
                        parts = []
                        for i in range(arity):
                            x, y = left[i], right[perm[i]]
                            parts.append(x | y if i == union_at else x & y)
                        if all(parts):
                            generated.add(_canon_box(parts))
        nxt = _maximal(generated)
        if nxt == current:
            return nxt
        current = nxt


def _label_key(label: Any) -> str:
    if isinstance(label, str):
        return json.dumps(label, ensure_ascii=False)
    if not isinstance(label, list) or not label:
        raise ValueError("labels must be strings or nonempty nested subset arrays")
    # A subset is a set at every nesting level, not a multiset.
    return "[" + ",".join(sorted({_label_key(member) for member in label})) + "]"


def _parse_problem(raw: dict[str, Any]):
    if not isinstance(raw, dict):
        raise ValueError("problem must be a JSON object")
    alphabet = raw.get("alphabet", raw.get("labels"))
    if not isinstance(alphabet, list):
        raise ValueError("alphabet must be an array of labels")
    encoded_alphabet = [_label_key(x) for x in alphabet]
    if len(set(encoded_alphabet)) != len(encoded_alphabet):
        raise ValueError("alphabet labels must be unique")
    allowed = set(encoded_alphabet)

    def constraint(name: str, aliases: tuple[str, ...]):
        value = next((raw[key] for key in (name, *aliases) if key in raw), None)
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array of configurations")
        out = set()
        for conf in value:
            if not isinstance(conf, list) or not conf:
                raise ValueError(f"each {name} configuration must be a nonempty label array from alphabet")
            encoded = tuple(sorted(_label_key(x) for x in conf))
            if any(x not in allowed for x in encoded):
                raise ValueError(f"each {name} configuration must use labels from alphabet")
            out.add(encoded)
        arities = {len(c) for c in out}
        if len(arities) > 1:
            raise ValueError(f"all {name} configurations must have equal size")
        arity_key = "node_arity" if name == "node_constraint" else "edge_arity"
        arity = raw.get(arity_key, next(iter(arities), None))
        if type(arity) is not int or arity < 1:
            raise ValueError(f"{arity_key} must be a positive integer for an empty constraint")
        if arities and arities != {arity}:
            raise ValueError(f"{arity_key} does not match {name} configuration length")
        return out, arity

    nodes, node_arity = constraint("node_constraint", ("node", "white"))
    edges, edge_arity = constraint("edge_constraint", ("edge", "black"))
    return nodes, edges, node_arity, edge_arity


def _existential_constraint(candidates: set[Label], valid: set[tuple[str, ...]]) -> set[Box]:
    # Generate choices containing a witness instead of testing every possible
    # multiset of candidates. Canonicalization accounts for every permutation
    # of the witness without explicitly enumerating those permutations.
    containing = {label: sorted((part for part in candidates if label in part),
                                key=lambda s: (len(s), tuple(sorted(s))))
                  for conf in valid for label in conf}
    return {_canon_box(choice) for conf in valid
            for choice in itertools.product(*(containing[label] for label in conf))}


def _serialize_box(box: Box) -> list[list[Any]]:
    return [[json.loads(x) for x in sorted(part)] for part in box]


def _box_key(box: Box):
    return tuple((len(part), tuple(sorted(part))) for part in box)


def round_eliminate(problem: dict[str, Any], operator: str = "R") -> dict[str, Any]:
    """Apply R or Rbar, matching configurations as unordered multisets."""
    if operator not in {"R", "Rbar"}:
        raise ValueError("operator must be 'R' or 'Rbar'")
    nodes, edges, node_arity, edge_arity = _parse_problem(problem)
    universal = _newre(edges if operator == "R" else nodes,
                       edge_arity if operator == "R" else node_arity)
    candidates = {part for box in universal for part in box}
    other = _existential_constraint(candidates, nodes if operator == "R" else edges)
    new_nodes, new_edges = (other, universal) if operator == "R" else (universal, other)
    output = {
        "alphabet": [[json.loads(x) for x in sorted(s)] for s in sorted(candidates, key=lambda s: (len(s), tuple(sorted(s))))],
        "node_constraint": [_serialize_box(b) for b in sorted(new_nodes, key=_box_key)],
        "edge_constraint": [_serialize_box(b) for b in sorted(new_edges, key=_box_key)],
    }
    if not new_nodes:
        output["node_arity"] = node_arity
    if not new_edges:
        output["edge_arity"] = edge_arity
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="input JSON file (default: stdin)")
    parser.add_argument("--output", "-o", default="RE_out.json",
                        help="JSON output path (default: RE_out.json)")
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--operator", choices=("R", "Rbar"), default="R",
                            help="one paper round-elimination operator (default: R)")
    operations.add_argument("--two-step", action="store_true",
                            help="apply Rbar(R(problem)), the standard full speed-up step")
    args = parser.parse_args()
    try:
        input_context = (open(args.input, encoding="utf-8") if args.input not in (None, "-")
                         else nullcontext(sys.stdin))
        with input_context as stream:
            problem = json.load(stream)
        if args.two_step:
            output = round_eliminate(round_eliminate(problem, "R"), "Rbar")
        else:
            output = round_eliminate(problem, args.operator)
        output_context = (open(args.output, "w", encoding="utf-8") if args.output != "-"
                          else nullcontext(sys.stdout))
        with output_context as stream:
            json.dump(output, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"round_eliminator: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
