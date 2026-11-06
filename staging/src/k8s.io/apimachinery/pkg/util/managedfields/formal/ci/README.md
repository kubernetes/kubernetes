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
four imported function ASTs must match the checked-in Lean input.

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
8. Actual merge tests and exact Lean/Go comparisons on both pinned SMD PR #336
   head and base revisions (Go may install the upstream-required toolchain).
9. Reproducibility of the self-contained HTML proof explorer against the Lean
   environment export and source snapshots.
10. Four independent semantic mutations of the selected Go source; each must
    invalidate the unchanged theorem in the expected proof module.

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
