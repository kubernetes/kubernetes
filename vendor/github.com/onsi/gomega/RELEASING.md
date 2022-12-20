A Gomega release is a tagged sha and a GitHub release.  To cut a release:

1. Ensure CHANGELOG.md is up to date.
  - Use 
    ```bash
    LAST_VERSION=$(git tag --sort=version:refname | tail -n1)
    CHANGES=$(git log --pretty=format:'- %s [%h]' HEAD...$LAST_VERSION)
    echo -e "## NEXT\n\n$CHANGES\n\n### Features\n\n## Fixes\n\n## Maintenance\n\n$(cat CHANGELOG.md)" > CHANGELOG.md
    ```
   to update the changelog
  - Categorize the changes into
    - Breaking Changes (requires a major version)
    - New Features (minor version)
    - Fixes (fix version)
    - Maintenance (which in general should not be mentioned in `CHANGELOG.md` as they have no user impact)
1. Update GOMEGA_VERSION in `gomega_dsl.go`
1. Commit, push, and release:
  ```
  git commit -m "vM.m.p"
  git push
  gh release create "vM.m.p"
  git fetch --tags origin master
  ```