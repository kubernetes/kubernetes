A Ginkgo release is a tagged git sha and a GitHub release.  To cut a release:

1. Ensure CHANGELOG.md is up to date.
  - Use `git log --pretty=format:'- %s [%h]' HEAD...vX.X.X` to list all the commits since the last release
  - Categorize the changes into
    - Breaking Changes (requires a major version)
    - New Features (minor version)
    - Fixes (fix version)
    - Maintenance (which in general should not be mentioned in `CHANGELOG.md` as they have no user impact)
1. Update `VERSION` in `types/version.go`
1. Commit, push, and release:
  ```
  git commit -m "vM.m.p"
  git push
  gh release create "vM.m.p"
  git fetch --tags origin master
  ```