#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SSA_LAKE="${SSA_LAKE:-lake}"
mkdir -p .lake/audit-checks

function expect-rejection {
  local name="$1"
  local diagnostic="$2"
  local probe=".lake/audit-checks/$name.lean"
  local output=".lake/audit-checks/$name.log"
  if "${SSA_LAKE}" env lean "$probe" >"$output" 2>&1; then
    printf 'Audit unexpectedly accepted %s; see %s\n' "$name" "$output" >&2
    exit 1
  fi
  if ! grep -Fq "$diagnostic" "$output"; then
    cat "$output" >&2
    printf 'Audit failed for the wrong reason in %s\n' "$name" >&2
    exit 1
  fi
  printf 'Audit correctly rejected %s.\n' "$name"
}

# These probes must fail even when the offending declaration is not in a
# hand-maintained export list or when elaboration warnings are suppressed.
cat >.lake/audit-checks/extra-axiom.lean <<'LEAN'
import SSA.Audit
axiom unexpectedAssumption : False
audit_ssa
LEAN
expect-rejection extra-axiom 'forbidden axiom'

cat >.lake/audit-checks/hidden-sorry.lean <<'LEAN'
import SSA.Audit
set_option warningAsError false
private theorem incompleteDependency : False := by sorry
theorem incompleteExport : False := incompleteDependency
audit_ssa
LEAN
expect-rejection hidden-sorry 'forbidden axiom sorryAx'

cat >.lake/audit-checks/native-evaluation.lean <<'LEAN'
import SSA.Audit
theorem uncheckedEvaluation : 2 + 2 = 4 := by native_decide
audit_ssa
LEAN
expect-rejection native-evaluation 'forbidden axiom'

outside="OutsideNamespaceAuditProbe.lean"
unimported="SSA/UnimportedAuditProbe.lean"
if [[ -e "$outside" || -e "$unimported" ]]; then
  printf 'Refusing to overwrite an existing audit coverage probe source.\n' >&2
  exit 1
fi
trap 'rm -f "$outside" "$unimported"' EXIT
printf 'axiom assumptionOutsideSSA : False\n' >"$outside"
"${SSA_LAKE}" env lean -o .lake/build/lib/lean/OutsideNamespaceAuditProbe.olean "$outside" \
  >.lake/audit-checks/outside-module-build.log 2>&1
cat >.lake/audit-checks/outside-namespace.lean <<'LEAN'
import SSA.Audit
import OutsideNamespaceAuditProbe
audit_ssa
LEAN
expect-rejection outside-namespace 'forbidden axiom'
rm -f "$outside"

printf 'namespace SSA\nend SSA\n' >"$unimported"
cat >.lake/audit-checks/missing-module.lean <<'LEAN'
import SSA.Audit
audit_ssa
LEAN
expect-rejection missing-module 'is not imported by SSA.lean'
