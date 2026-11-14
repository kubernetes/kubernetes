# SSA verification in CI

Run from any directory:

```sh
bash /path/to/apimachinery/pkg/util/managedfields/formal/ci/verify.sh
```

The same entrypoint works in the Kubernetes staging checkout and in the
published apimachinery repository. Go and Python 3 must already be installed;
the module's Go requirements apply. Downloading Lean requires curl, tar with zstd support,
and either sha256sum or shasum. Lean is not installed globally.

The script selects the release pinned by `../lean-toolchain`, verifies its
archive against `lean.sha256`, and caches the extracted toolchain. The checksums
come from the assets of the official
[Lean 4.35.0 release](https://github.com/leanprover/lean4/releases/tag/v4.35.0).
Toolchain updates must update both files. `SSA_TOOLCHAIN_CACHE` changes the
cache directory; `SSA_LAKE` selects an existing installation, whose version is
still checked. `ARTIFACTS` selects the log directory, matching Prow conventions.
Without it, a process-specific directory under the temporary directory is used.
Go locates the selected SMD dependency's source, whether it is in Kubernetes'
vendor directory or a standalone module cache. Its full source digest and all
six updater ASTs, three element-set algebra ASTs and five recursive Difference/filter
ASTs, two Has/Less ASTs, two typed map/list field-extraction ASTs, four atomic merge/leaf/rule ASTs, four structural removal/extraction/caller ASTs, three schema-selection ASTs, three schema lookup/cache/copy ASTs and the selected Go standard-library search AST must match the checked-in Lean input, including all nine complete
source-file digests.

Native overlays run against the selected vendor source directly. Go rejects
overlays below GOMODCACHE, so `run-native-go.sh` copies a cached SMD module to an
isolated temporary directory, remaps only that module's overlay targets and
uses the same selected Go compiler with the copied module graph read-only.
Only copied directory permissions are changed for cleanup; original cache
sources and module/workspace files remain untouched. The standard-library
mutation keeps its separate isolated GOROOT mirror. `GOWORK=off` exercises the
module-cache selection used by standalone apimachinery.

The source schema overlay additionally compares 5,120 real plain/full lookups
with the source interpreter, covering cache allocation/hits/misses, repeated
calls, copied and unchanged pointers, all opaque exported container contents,
map field-index initialization and the actual unlocked mutex. Both pinned
versions run that matrix and a 64-goroutine control; the current dependency also
runs the control with `-race`. Five mutations check missing-name failure,
override assignment, cache-hit polarity, deferred unlock and writes through an
escaped local copy. The whole source suite contains 33 semantic mutations,
25 also rejected by an independent native oracle.

The complete Map.CopyInto AST additionally matches 252 direct native copies,
including nil destinations, nil/empty/populated cache pointers and independent
Once tokens. A reused, normally indexed destination reproduces the stale-index
counterexample; Resolve uses a fresh Map. Two further mutations remove Fields
assignment and Once reset, requiring both unchanged source and native rejection.

The required checks are:

1. Clean Lake build with warnings as errors.
2. Automatic transitive axiom audit of all model declarations. Every project
   source module must occur in the audited import graph.
3. Fresh kernel replay using the toolchain's built-in `leanchecker`.
4. Negative audit tests for an extra axiom, a hidden incomplete dependency,
   native evaluation, an imported module outside the SSA namespace, and an
   omitted source module.
5. Kernel-checked examples, executable scenario replay, and fixture regeneration.
6. Ordinary Go tests against independently authored YAML outcomes, including
   actual FieldsV1 ownership serialization between requests.
7. Live Lean/Go comparison of the corpus and deterministic generated sequences.
   The imported element-set interpreter also runs against real Go and an
   independent membership oracle on all 4,096 subset pairs per operation
   (12,288 total), covering field, key, value and index selectors.
   Separate numeric witnesses compare imported Apply/element-set execution
   with real mixed numeric equality and managedfields, including both false
   no-op and missed-conflict outcomes. Reader tests preserve exact int64 tokens
   in requests, independent expectations, paths and schema defaults.
   Numeric FieldsV1 selectors are also compared by exact native magnitude and
   complete reserialized bytes, independently of the lossy `Has`/`Equals` oracle.
   The imported Apply counterexample additionally checks independent-key false
   conflict, unchanged-payload ownership loss with force, and small-key success.
   An internal-package Go overlay also compares full imported ownership-update
   observations: shared comparison-cache filtering and forced removal of B's
   conflicting claim without acquiring the actor's newly added path.
   The complete recursive Difference source also matches 4,102 native tree
   pairs, including exact large numeric tokens, direct/child overlap, empty
   structural nodes and depth 64. Another 518 pairs check conventional root
   reset filters and preservation of independent prefixes. Has/standard-library
   search add 776 source/native queries, 128 Less results and 16 near-limit
   32/64-bit unsigned-midpoint controls.
   Typed atomic field extraction adds 320 source/native map/list cases, covering
   null/empty/deep/wide values, empty-root behavior, prior fields and deferred
   cleanup counts. Both pinned versions also run the full typed suite and overlay.
   Another 2,352 cases compare the complete atomic map/list merge dispatch,
   leaf guard and actual rule closure: exact whole operands, missing/null,
   previous leaf/output, numeric/wide/deep data, scope immutability and LIFO frees.
   Removal/extraction adds 1,152 complete-helper comparisons: both modes, scalar
   and atomic map/list dispatch, prior output, raw nil versus typed nil, empty,
   numeric, wide/deep values, selection-set immutability and captured cleanup.
   Another 1,152 complete-caller comparisons derive fresh initialization, selected
   handler composition, actual valid/missing/invalid schema resolution and null
   wrapping after normal resolver errors; exact input/selection state is checked.
   These caller cases also run the full source-derived schema-selection chain.
   Schema helpers additionally compare 1,296 deductions, 96 exact handler calls
   and 7,776 complete resolver calls: all pointer masks and predicate combinations,
   nil/null, real typed values, actual lookup boundaries, callback errors and prior
   state. Independent bit-mask/priority and pointer-identity oracles check results.
8. Actual merge and fieldpath tests and exact Lean/Go comparisons on both pinned SMD PR #336
   head and base revisions (Go may install the upstream-required toolchain).
   Each revision also runs the numeric-key round-trip counterexample and control.
9. Reproducibility of the self-contained HTML proof explorer against the Lean
   environment export and source snapshots.
10. Twenty-six independent semantic mutations of the selected Go sources; each must
    invalidate the unchanged theorem in the expected proof module.
    The ownership-growth mutation must additionally fail the native Go witness
    for its demonstrated new B claim, rather than a compile or setup error.
    A matched-child representative mutation must independently invent an exact
    native path and fail the unchanged recursive source certificate. A reset
    mutation using ordinary Difference must fail both the source filter proof
    and the native whole-prefix exclusion oracle. Flipping the selected
    standard-library search’s final zero test must fail both its source proof
    and the independent native membership oracle.
    Independent map/list atomic-guard reversals must fail the typed source
    certificate and the native whole-scope claim oracle, including null input.
    Clearing the chosen right output and independently bypassing the map/list
    leaf guard must each fail the unchanged atomic merge certificate and native
    whole-operand/call-frame oracle. Clearing scalar removal output and inverting
    each atomic extraction guard must fail unchanged structural-source theorems
    and the independent native whole-output oracle. The constructor extract flag,
    skipped schema call and discarded caller result must likewise fail the source
    caller theorem and its independent native output oracle. Eighteen mutations have native
    checks. Whole-atom scalar deduction, swapped handler priority and skipped
    deduction must fail their unchanged source certificate and independent native
    deduction/handler/error oracle. Compile/setup failures do not count.

No check is optional when invoking this entrypoint. The result establishes the
documented proof and comparison checks, not correctness of the human-chosen
formalization. Proof boundaries are listed in `../PROPERTIES.md`.

## Prow configuration

`presubmits.yaml` is a concrete proposed
`config/jobs/kubernetes/sig-api-machinery/ssa-formal.yaml` in
`kubernetes/test-infra`. It uses the existing kubekins image tag, requests a
blocking job for changes to apimachinery and relevant SMD/dependency files, and
invokes this entrypoint followed by `verify-apiserver.sh`. It needs a separate test-infra change and the normal
job review/activation; Kubernetes does not discover Prow jobs from this directory.
The Linux Prow job has not been run by the local macOS verification.

The bootstrap may download about 600 MB on a cold cache. A dedicated image with
the same pinned Lean release can instead set `SSA_LAKE`. The source checkout is
always rebuilt, including when the toolchain itself is cached.

## Kubernetes HTTP and storage regression

```sh
bash ci/verify-apiserver.sh
```

This requires a Kubernetes checkout. It uses the checkout's normal etcd
installation and lifecycle helpers, starts apiextensions-apiserver, and runs
`TestSSAEmptyAssociativeListPresence` through HTTP and etcd. The test checks
optional-list data non-idempotence, the default wrapper's synthetic Update
claim, Required/CEL rejection of otherwise valid empty-list intent, and atomic
rollback verified by GET, including ownership and `resourceVersion`. Complete
API-server and etcd logs are retained under `ARTIFACTS`.

The same entrypoint also runs `TestSSAEmptyGranularMapPresence`: optional
non-nullable rejection, nullable null/empty divergence, identical-intent
conflict, CEL rejection, an empty-object default and an atomic-map control.

`TestSSAUpdateOwnedListItemPruning` checks the separate Update/Apply handoff:
B Updates a scalar child without acquiring item membership; A's omission prunes
the item. A nullable list stores null and loses the claims, while an optional
non-nullable list rejects every retry with unchanged data, ownership and
resourceVersion. Empty creation without the optional list is a valid control.
Both policies also run after A reasserts B's unchanged value, with exact
FieldsV1 checks proving genuine A/B scalar co-ownership before omission.
Accepted deletions are checked by GET; the shared-scalar variants reject the
unrestricted BASE-03 guarantee as well as the original BASE-07 reading.

`TestSSAStatusResetOwnership` checks same-name main/status manager separation
and conventional spec/status reset behavior. Its additional-root variant
reproduces a status claim on an unchanged `settings.desired` field that the REST
strategy resets; the original main manager's next change conflicts atomically.

`TestSSANumericComparison` checks a valid optional `type: number` scalar with
no defaults, validation rule or webhook. A owns MaxInt64; B requests binary64
`2^63` without force. Scalar-only intent is lost as a false no-op; with an
unrelated note the different number persists while A retains ownership. A
different exact int64 correctly conflicts. GET checks every accepted result
and rejection.
Two forced variants additionally check the retained foreign claim on the
rounded change and the successful transfer for an exact integer change. Both
outcomes also have source-model proofs and native managedfields comparisons.

`TestSSANumericListKeyOwnership` checks an integer-keyed map list. A's exact
`2^53+1` key survives in stored data and raw FieldsV1, but decoding ownership
rounds it to `2^53`; B adding that distinct neighbor item falsely conflicts.
GET verifies unchanged data, claims and resourceVersion. Small independent
integer keys succeed and preserve both items.
The large-key branch also forces the rejected request: both items and the old
payload persist, but A's old claims disappear. Exact B key ownership and GET
persistence are checked. This adds a transition to the existing variant.

`TestSSANumericListKeyOmission` checks two adjacent keys with and without B
co-applying the upper payload. A's lower payload is exclusively A-owned in the
exact raw FieldsV1 before omission. Both item keys stay asserted. Large shared
keys retain the omitted payload and give its claim to B after decoding; all
three controls remove it. GET matches the response. These seven functions
exercise twenty-five API/schema variants. The four omission observations
also agree with Lean in native vendored and both pinned SMD executions.

The standalone apimachinery proof entrypoint has no dependency on this
Kubernetes-only test. The proposed Kubernetes Prow job requires both checks;
the ordinary apiextensions integration suite also discovers the HTTP test.

## Run only the implementation comparison during development

After building the Lean executable:

```sh
go test ./bridge ./cmd/ssa-compare
go run ./cmd/ssa-compare -model .lake/build/bin/ssa-model
go run ./cmd/ssa-compare -model .lake/build/bin/ssa-model -seed 42 -generated 128
go run ./cmd/ssa-compare -model .lake/build/bin/ssa-model -reproducer failure.json
```

Each hand-authored step has a complete expected object and owner map. Expected
proposal differences require both the Lean and Go outcome; changes to either
outcome fail. No model-defect exception category is accepted. Generated families
cover scalars, structural Apply/Update, and the default Apply bootstrap wrapper.
The bootstrap family is Apply-only; probabilistic non-Apply tracking is rejected
as outside the model rather than silently approximated.
The normal corpus reader refuses missing golden outcomes; only the explicit
reproducer mode permits generated prefixes with no golden outcomes.

The observation projection drops the API type envelope, redundant list-key leaf
claims. It compares all object cells, empty
containers, scalar/atomic ownership, item membership, and sparse container presence ownership. Associative-list
ordering is outside the comparison. The bridge uses the real Go merge/pruning
logic. Its schema conversion and observation
projection require human review.

## Pinned PR comparison

```sh
bash ci/verify-pr336.sh
```

The script fetches immutable upstream SHAs into the ignored `.lake/` cache,
asserts their identity, and runs both merge suites plus the exact comparison.
Generated harness sources reuse this project's independent schema walker and
Kubernetes' pinned YAML decoder; no module file or upstream implementation is
modified. `GOPATH` and `GOMODCACHE` can select writable task-local Go caches.
The primary entrypoint includes this check; it is not optional in full verification.
