<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/devel/cherry-picks.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Overview

This document explains cherry picks are managed on release branches within the
Kubernetes projects.  Patches are either applied in batches or individually
depending on the point in the release cycle.

## Propose a Cherry Pick

### BATCHING: After branching during code slush up to X.X.0

Starting with the release-1.2 branch, we shifted to a new cherrypick model
where the branch 'OWNERS' cherry pick batches of patches into the branch
to control the order and also vet what is or is not cherrypicked to a branch.

Contributors that want to include a cherrypick for a branch should label
their PR with the `cherrypick-candidate` label **AND** mark it
with the appropriate milestone (or the bot will unlabel it).

These cherrypick-candidate's will be triaged, batched up and submitted
to the release branch by the branch OWNERS.

There is an [issue](https://github.com/kubernetes/kubernetes/issues/23347) open to automate this new procedure.

### INDIVIDUAL CHERRYPICKS: Post X.X.0

```sh
hack/cherry_pick_pull.sh upstream/release-3.14 98765
```

This will walk you through the steps to propose an automated cherry pick of pull
 #98765 for remote branch `upstream/release-3.14`.

#### Cherrypicking a doc change

If you are cherrypicking a change which adds a doc, then you also need to run
`build/versionize-docs.sh` in the release branch to versionize that doc.
Ideally, just running `hack/cherry_pick_pull.sh` should be enough, but we are not there
yet: [#18861](https://github.com/kubernetes/kubernetes/issues/18861)

To cherrypick PR 123456 to release-1.2, run the following commands after running `hack/cherry_pick_pull.sh` and before merging the PR:

```
$ git checkout -b automated-cherry-pick-of-#123456-upstream-release-1.2
  origin/automated-cherry-pick-of-#123456-upstream-release-1.2
$ ./build/versionize-docs.sh release-1.2
$ git commit -a -m "Running versionize docs"
$ git push origin automated-cherry-pick-of-#123456-upstream-release-1.2
```

## Cherry Pick Review

Cherry pick pull requests are reviewed differently than normal pull requests. In
particular, they may be self-merged by the release branch owner without fanfare,
in the case the release branch owner knows the cherry pick was already
requested - this should not be the norm, but it may happen.

## Searching for Cherry Picks

See the [cherrypick queue dashboard](http://cherrypick.k8s.io/#/queue) for
status of PRs labeled as `cherrypick-candidate`.

[Contributor License Agreements](http://releases.k8s.io/HEAD/CONTRIBUTING.md) is considered implicit
for all code within cherry-pick pull requests, ***unless there is a large
conflict***.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/cherry-picks.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
