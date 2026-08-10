# Versioning and Release

The `release.json` file at the root of the repo determines which release line
this branch belongs to.

```js
{
  "majorVersion": 1 | 2,
  "releaseType": "stable" | "alpha" | "rc"
}
```

To reduce merge conflicts in automatic merges between version branches, the
current version number is stored under `version.vNN.json` (where `NN` is
`majorVersion`) and changelogs are stored under `CHANGELOG.NN.md` (for
historical reasons, the changelog for 1.x is under `CHANGELOG.md`). When we
fork to a new release branch (e.g. `v2-main`), we will update `release.json` in
this branch to reflect the new version line, and this information will be used
to determine how releases are cut.

The actual `version` field in all `package.json` files should always be `0.0.0`.
This means that local development builds will use version `0.0.0` instead of the
official version from the version file.

## `./bump.sh`

This script uses [standard-version] to update the version in `version.vNN.json`
to the next version. By default it will perform a **minor** bump, but `./bump.sh
patch` can be used to perform a patch release if that's needed.

This script will also update the relevant changelog file.

[standard-version]: https://github.com/conventional-changelog/standard-version

## `scripts/resolve-version.js`

The script evaluates the configuration in `release.json` and exports an
object like this:

```js
{
  version: '2.0.0-alpha.1',          // the current version
  versionFile: 'version.v2.json',    // the version file
  changelogFile: 'CHANGELOG.v2.md',  // changelog file name
  prerelease: 'alpha',               // prerelease tag (undefined for stable)
  marker: '0.0.0'                    // version marker in package.json files
}
```

## scripts/align-version.sh

In official builds, the `scripts/align-version.sh` is used to update all
`package.json` files based on the version from `version.vNN.json`.