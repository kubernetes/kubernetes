# Releasing Ginkgo

A release is one GitHub Actions run: **Actions → Release → Run workflow → bump: patch | minor**,
from `master`. From the command line:

```bash
gh workflow run release.yml -f bump=patch   # or -f bump=minor
```

The run:

1. Runs the whole test workflow (`test.yml`). Nothing else happens unless it passes.
2. Runs `scripts/release.sh`, which
   - computes the next version from `VERSION` in `types/version.go`,
   - renames `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z` (dropping empty `###` subsections) and
     puts a fresh, empty `## Unreleased` above it,
   - writes `VERSION` and stamps the version into every plugin manifest
     (`plugins/*/.claude-plugin/plugin.json`), so the Claude Code plugin ships in lockstep,
   - commits `vX.Y.Z` as `github-actions[bot]`, tags it `vX.Y.Z`, and pushes the commit and the tag
     together (`git push --atomic`, fast-forward only),
   - creates the GitHub release `vX.Y.Z`, whose notes are the body of `## X.Y.Z`.

The Go module is released by the `vX.Y.Z` tag. There is no other release tooling.

## The changelog

`CHANGELOG.md` starts with an `## Unreleased` section:

```markdown
## Unreleased

### Features

### Fixes

### Maintenance

## 2.32.2
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
| The push (master moved during the run) | The re-run fails the same way. Start a new run from the current master. |
| After the push (the tag `vX.Y.Z` is on GitHub) | The re-run builds from the tag instead of making a new commit. |
| Creating the GitHub release | The release is created, or its notes are refreshed if it already exists. |

The commit and the tag are pushed together, so the tag being on GitHub means the release commit is
on master too.

## Trying the release script locally

`GINKGO_RELEASE_DRY_RUN=1 scripts/release.sh patch` does the whole release locally — the changelog
and version rewrites, the commit and the tag — then skips the push and the GitHub release. It still
checks `origin` for an existing tag, so run it in a throwaway clone whose `origin` is a local bare
repository, not in your working copy.

The version and changelog logic lives in `scripts/release`, with its own Ginkgo suite
(`scripts/release/release_test.go`) that runs with the rest of the suites.
