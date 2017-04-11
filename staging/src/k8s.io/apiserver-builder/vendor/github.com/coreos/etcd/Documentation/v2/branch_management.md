# Branch Management

## Guide

* New development occurs on the [master branch][master].
* Master branch should always have a green build!
* Backwards-compatible bug fixes should target the master branch and subsequently be ported to stable branches.
* Once the master branch is ready for release, it will be tagged and become the new stable branch.

The etcd team has adopted a *rolling release model* and supports one stable version of etcd.

### Master branch

The `master` branch is our development branch. All new features land here first.

If you want to try new features, pull `master` and play with it. Note that `master` may not be stable because new features may introduce bugs.

Before the release of the next stable version, feature PRs will be frozen. We will focus on the testing, bug-fix and documentation for one to two weeks.

### Stable branches

All branches with prefix `release-` are considered _stable_ branches.

After every minor release (http://semver.org/), we will have a new stable branch for that release. We will keep fixing the backwards-compatible bugs for the latest stable release, but not previous releases. The _patch_ release, incorporating any bug fixes, will be once every two weeks, given any patches.

[master]: https://github.com/coreos/etcd/tree/master
