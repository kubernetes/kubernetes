# Releasing Gomega

A release is one GitHub Actions run: **Actions → Release → Run workflow → bump: patch | minor**,
from `master`. From the command line:

```bash
gh workflow run release.yml -f bump=patch   # or -f bump=minor
```

The run:

1. Runs the whole test workflow (`test.yml`). Nothing else happens unless it passes.
2. Runs `scripts/release.sh`, which
   - computes the next version from `GOMEGA_VERSION` in `gomega_dsl.go`,
   - renames `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z` (dropping empty `###` subsections) and
     puts a fresh, empty `## Unreleased` above it,
   - writes `GOMEGA_VERSION` and stamps the version into every plugin manifest
     (`plugins/*/.claude-plugin/plugin.json`), so the Claude Code plugin ships in lockstep,
   - commits that on `master` as `vX.Y.Z (full)`, as `github-actions[bot]`,
   - builds the released tree on `master-lite` and tags **that** commit `vX.Y.Z` (see below),
   - pushes `master`, `master-lite`, and the tag together (`git push --atomic`, fast-forward only),
   - creates the GitHub release `vX.Y.Z`, whose notes are the body of `## X.Y.Z`.

## master and master-lite

Gomega uses Ginkgo for its own tests, and the go toolchain pulls a dependency's *test* dependencies
into consuming projects as indirect requirements — so a project that uses only Gomega would end up
with all of Ginkgo in its `go.mod`. Since 1.40.0, releases therefore ship a stripped-down tree:

- `master` holds the real repository, tests and all. The release commit there is `vX.Y.Z (full)`,
  and it is **not** what the go toolchain resolves.
- `master-lite` holds the released trees: the same tree with every `_test.go` file deleted and
  `go.mod` re-tidied, which drops Ginkgo. `scripts/strip-tests.sh` does that, and asserts that the
  result still builds and no longer requires Ginkgo.
- Each release adds one commit to `master-lite`: a merge whose tree is the stripped tree and whose
  parents are the previous `master-lite` commit and the `vX.Y.Z (full)` commit on `master`. So
  `master-lite`'s history connects to master's, and `git log master-lite` reads as the release
  history.
- The `vX.Y.Z` tag is on that `master-lite` commit. It is what `go get github.com/onsi/gomega`
  resolves, so `go build` never even sees the test files.

The test workflow's **Check the released (lite) tree** job runs `scripts/strip-tests.sh` on every
push. If a non-test file ever imports Ginkgo, that job fails long before release day — the released
module could not be built otherwise.

Release tags are not on `master`, so `git describe` on master does not find them. `GOMEGA_VERSION`
in `gomega_dsl.go` is the version of record, and `git tag --sort=version:refname | tail -n1` names
the latest release.

If this release process ever causes unexpected changes for a project, please open an issue.

### Never merge master-lite into master

`master-lite` always looks ahead of `master`, because its newest commit merges master's release
commit. But its *tree* has the tests taken out, so merging or rebasing it back into master deletes
every `_test.go` file and drops Ginkgo from `go.mod` — cleanly, with no conflicts. GitHub's
"master-lite had recent pushes — Compare & pull request" banner is the one click to avoid.

Nothing you do locally needs `master-lite`: the release workflow writes it from a fresh checkout. A
stale local `master-lite` branch is harmless, and you can delete it.

`scripts/check-full-tree.sh` guards against a stripped master: it asserts that a checkout has
`_test.go` files and that `go.mod` requires Ginkgo. The test workflow runs it on every push and
`scripts/release.sh` runs it before cutting a release, so a stripped master gets reported plainly
instead of turning into a test run with nothing in it.

### Cutting one release after another

Work on master as usual between releases. The `vX.Y.Z (full)` commit the workflow pushed is an
ordinary commit: `git pull` fast-forwards onto it, it carries the released `CHANGELOG.md` and
`GOMEGA_VERSION`, and you add new `## Unreleased` entries above it. The next release repeats the
same two steps — a new `vX.Y.Z (full)` commit on master, and a new merge on `master-lite` whose
tree is the stripped version of it, carrying the tag.

Nothing accumulates between releases, and nothing needs reconciling in either direction: each
`master-lite` commit's tree is built from the release commit rather than merged into the previous
one, so `git diff vX.Y.Z vX.Y.Z+1` over two releases shows only the real changes.

## The changelog

`CHANGELOG.md` starts with an `## Unreleased` section:

```markdown
## Unreleased

### Features

### Fixes

### Maintenance

## 1.43.0
...
```

Add an entry under `## Unreleased` with each user-facing change, as you make it. Never edit a
released section. Empty subsections are dropped when the section is released, so leaving
`### Maintenance` empty costs nothing.

If `## Unreleased` has no entries (only blank lines and `###` headings), the release stops before
changing anything — and so does a re-run after a successful release.

To draft entries from the commits since the last release:

```bash
git log --pretty=format:'- %s [%h]' "$(git tag --sort=version:refname | tail -n1)"..HEAD
```

Categorize what you keep into

- Breaking changes (these need a major version, which this workflow does not cut)
- `### Features` (minor version)
- `### Fixes` (patch version)
- `### Maintenance` (in general, changes with no user impact don't belong in the changelog at all)

## If a release fails

Open the failed run and click **Re-run failed jobs**. A re-run checks out the same commit as the
first attempt and computes the same version. It resumes based on what already exists:

| Where it failed | What a re-run does |
|---|---|
| Tests, or before the push | Nothing reached GitHub. The re-run starts over. |
| The push (master or master-lite moved during the run) | The re-run fails the same way. Start a new run from the current master. |
| After the push (the tag `vX.Y.Z` is on GitHub) | The re-run builds from the tag instead of making new commits. |
| Creating the GitHub release | The release is created, or its notes are refreshed if it already exists. |

Both branches and the tag are pushed together, so the tag being on GitHub means both release
commits are there too.

## Trying the release script locally

`GOMEGA_RELEASE_DRY_RUN=1 scripts/release.sh patch` does the whole release locally — the changelog
and version rewrites, the `master` commit, the stripped `master-lite` commit and the tag on it —
then skips the push and the GitHub release. It still checks `origin` for an existing tag and needs
`origin/master-lite`, so run it in a throwaway clone whose `origin` is a local bare repository, not
in your working copy.

`scripts/strip-tests.sh` rewrites the working tree in place, so run *it* only on a throwaway
checkout too.

The version and changelog logic lives in `scripts/release`, with its own Ginkgo suite
(`scripts/release/release_test.go`) that runs with the rest of the suites.
