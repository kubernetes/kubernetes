#!/usr/bin/env python3

# Copyright The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the standalone proof explorer from Lean's accepted declarations."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = re.compile(
    r"^(?:@\[[^\]\n]*\]\s*)*(?P<private>private )?(?P<kind>theorem|def|inductive|structure|abbrev)\s+(?P<name>[\w.]+)",
    re.MULTILINE,
)
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def source_index(path, source):
    namespace = re.search(r"^namespace ([\w.]+)", source, re.MULTILINE)
    namespace = namespace.group(1) + "." if namespace else ""
    declarations = list(DECLARATION.finditer(source))
    result = {}
    for index, match in enumerate(declarations):
        name = namespace + match["name"]
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(source)
        block = source[match.start():end].rstrip()
        block = re.sub(r"\n/--.*?-/\s*$", "", block, flags=re.DOTALL).rstrip()
        block = re.sub(r"\nend [\w.]+\s*$", "", block).rstrip()
        preceding = source[:match.start()].rstrip()
        comment = re.search(r"/--((?:(?!-/).)*)-/$", preceding, re.DOTALL)
        description = ""
        if comment:
            description = " ".join(comment.group(1).split())
        result[name] = {
            "file": path,
            "line": source.count("\n", 0, match.start()) + 1,
            "source": block,
            "description": description,
            "private": bool(match["private"]),
            "kind": "theorem" if match["kind"] == "theorem" else "definition",
        }
    return result


def build():
    export = ROOT / ".lake/proof-data.json"
    if not export.exists():
        raise ValueError("Run lake build and lake env lean SSA/Audit.lean before generating documentation.")
    accepted = json.loads(export.read_text())
    sources = {}
    digest = hashlib.sha256()
    for item in sorted(accepted["sources"], key=lambda item: item["path"]):
        actual = (ROOT / item["path"]).read_text()
        if actual != item["source"]:
            raise ValueError(f"Stale Lean export for {item['path']}; rerun lake env lean SSA/Audit.lean.")
        digest.update(item["path"].encode() + b"\0" + actual.encode())
        if item["path"] != "SSA/Audit.lean":
            sources.update(source_index(item["path"], actual))
    notes = json.loads((ROOT / "docs/notes.json").read_text())
    declarations = {}
    for item in accepted["declarations"]:
        if item["name"] in sources:
            # Compiler-generated declarations do not become extra advertised theorems.
            declarations[item["name"]] = {**item, **sources[item["name"]]}
    for name, source in sources.items():
        if source["kind"] == "theorem" and name not in declarations:
            raise ValueError(f"The source theorem {name} is absent from the accepted Lean export.")
    for name, item in declarations.items():
        item["dependencies"] = [dep for dep in item["dependencies"] if dep in declarations and dep != name]
        item["dependents"] = sorted(
            other for other, node in declarations.items() if name in node["dependencies"] and other != name
        )
        if item["kind"] == "theorem" and not set(item["axioms"]) <= ALLOWED_AXIOMS:
            raise ValueError(f"Forbidden axiom in {name}: {item['axioms']}")
        executions = {
            "SSA.Examples", "SSA.Counterexamples", "SSA.RawSourceCounterexamples", "SSA.ContainerCounterexamples",
            "SSA.HandoffCounterexamples", "SSA.ResetCounterexamples", "SSA.NumericCounterexamples",
            "SSA.NumericCodecCounterexamples",
            "SSA.GranularKeyCounterexamples",
        }
        default_scope = "execution" if item["module"] in executions else "general"
        if item["private"]:
            default_scope = "helper"
        if item["kind"] != "theorem":
            default_scope = "definition"
        item.update(notes.get(name, {}))
        item.setdefault("scope", default_scope)
        item.setdefault("title", name.rsplit(".", 1)[-1].replace("_", " ").capitalize())
        item.setdefault("meaning", item["description"])
    unknown_notes = set(notes) - set(declarations)
    if unknown_notes:
        raise ValueError(f"Notes refer to absent Lean declarations: {sorted(unknown_notes)}")
    fixtures = json.loads((ROOT / "fixtures/scenarios.json").read_text())
    data = {
        "toolchain": accepted["toolchain"],
        "sourceDigest": digest.hexdigest(),
        "scenarioCount": len(fixtures),
        "transitionCount": sum(len(fixture["steps"]) for fixture in fixtures),
        "nodes": sorted(declarations.values(), key=lambda item: (item["module"], item["line"])),
        "sourceRoot": "../",
    }
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Lean source and notes are data, never executable HTML.
    serialized = serialized.replace("<", r"\u003c").replace("&", r"\u0026")
    template = (ROOT / "docs/proof-explorer.template.html").read_text()
    if template.count("/*@@PROOF_DATA@@*/{}") != 1:
        raise ValueError("The proof explorer template must have exactly one data placeholder.")
    return template.replace("/*@@PROOF_DATA@@*/{}", serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in explorer is stale")
    args = parser.parse_args()
    target = ROOT / "docs/proofs.html"
    result = build()
    if args.check:
        if not target.exists() or target.read_text() != result:
            parser.error("docs/proofs.html is stale; run python3 docs/generate.py after the Lean build.")
        print("Proof explorer matches the accepted Lean declarations and source.")
    else:
        target.write_text(result)
        print(f"Generated {target} ({len(result.encode()):,} bytes).")


if __name__ == "__main__":
    main()
