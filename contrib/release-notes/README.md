## A simple tool that scrapes the github API to build a starting point for release notes.

### Usage
Find the rollup PR number that was most-recently merged into the previous .0 release.
 * Note: The most-recent PR included in the release, but not the PR that cuts the release.
 * Something like: LAST_RELEASE_PR=$(git log v0.19.0 | grep "Merge pull request" | head -1)
 * ... but check the log manually to confirm.

Find the rollup PR number that was most-recently merged into the current .0 release.
 * Something like: CURRENT_RELEASE_PR=$(git log v0.20.0 | grep "Merge pull request" | head -1)
 * ... but check the log manually to confirm.

You'll need to manually remove any PRs there were cherrypicked into the previous release's patch versions.

There are too many PRs for the tool to work without an api-token.  See https://github.com/settings/tokens to generate one."


```bash
${KUBERNETES_ROOT}/build/make-release-notes.sh --last-release-pr=<pr-number> --current-release-pr=<pr-number> --api-token=<github-api-token>

```



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/release-notes/README.md?pixel)]()
