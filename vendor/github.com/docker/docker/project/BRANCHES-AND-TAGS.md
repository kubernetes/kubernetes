Branches and tags
=================

Note: details of the release process for the Engine are documented in the
[RELEASE-CHECKLIST](https://github.com/docker/docker/blob/master/project/RELEASE-CHECKLIST.md).

# Branches

The docker/docker repository should normally have only three living branches at all time, including
the regular `master` branch:

## `docs` branch

The `docs` branch supports documentation updates between product releases. This branch allow us to
decouple documentation releases from product releases.

## `release` branch

The `release` branch contains the last _released_ version of the code for the project.

The `release` branch is only updated at each public release of the project. The mechanism for this
is that the release is materialized by a pull request against the `release` branch which lives for
the duration of the code freeze period. When this pull request is merged, the `release` branch gets
updated, and its new state is tagged accordingly.

# Tags

Any public release of a compiled binary, with the logical exception of nightly builds, should have
a corresponding tag in the repository.

The general format of a tag is `vX.Y.Z[-suffix[N]]`:

- All of `X`, `Y`, `Z` must be specified (example: `v1.0.0`)
- First release candidate for version `1.8.0` should be tagged `v1.8.0-rc1`
- Second alpha release of a product should be tagged `v1.0.0-alpha1`
