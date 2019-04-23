A Ginkgo release is a tagged git sha and a GitHub release.  To cut a release:

1. Ensure CHANGELOG.md is up to date.
  - Use `git log --pretty=format:'- %s [%h]' HEAD...vX.X.X` to list all the commits since the last release
  - Categorize the changes into
    - Breaking Changes (requires a major version)
    - New Features (minor version)
    - Fixes (fix version)
    - Maintenance (which in general should not be mentioned in `CHANGELOG.md` as they have no user impact)
1. Update `VERSION` in `config/config.go`
1. Create a commit with the version number as the commit message (e.g. `v1.3.0`)
1. Tag the commit with the version number as the tag name (e.g. `v1.3.0`)
1. Push the commit and tag to GitHub
1. Create a new [GitHub release](https://help.github.com/articles/creating-releases/) with the version number as the tag  (e.g. `v1.3.0`).  List the key changes in the release notes.
