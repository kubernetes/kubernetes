<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Overview

This document explains cherry picks are managed on release branches within the
Kubernetes projects.  Patches are either applied in batches or individually
depending on the point in the release cycle.

## Propose a Cherry Pick

1. Cherrypicks are [managed with labels and milestones]
(pull-requests.md#release-notes)
1. To get a PR merged to the release branch, first ensure the following labels
   are on the original **master** branch PR:
  * An appropriate milestone (e.g. v1.3)
  * The `cherrypick-candidate` label
1. If `release-note-none` is set on the master PR, the cherrypick PR will need
   to set the same label to confirm that no release note is needed.
1. `release-note` labeled PRs generate a release note using the PR title by
   default OR the release-note block in the PR template if filled in.
  * See the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md) for more
    details.
  * PR titles and body comments are mutable and can be modified at any time
    prior to the release to reflect a release note friendly message.

### How do cherrypick-candidates make it to the release branch?

1. **BATCHING:** After a branch is first created and before the X.Y.0 release
  * Branch owners review the list of `cherrypick-candidate` labeled PRs.
  * PRs batched up and merged to the release branch get a `cherrypick-approved`
label and lose the `cherrypick-candidate` label.
  * PRs that won't be merged to the release branch, lose the
`cherrypick-candidate` label.

1. **INDIVIDUAL CHERRYPICKS:** After the first X.Y.0 on a branch
  * Run the cherry pick script. This example applies a master branch PR #98765
to the remote branch `upstream/release-3.14`:
`hack/cherry_pick_pull.sh upstream/release-3.14 98765`
  * Your cherrypick PR (targeted to the branch) will immediately get the
`do-not-merge` label. The branch owner will triage PRs targeted to
the branch and label the ones to be merged by applying the `lgtm`
label.

There is an [issue](https://github.com/kubernetes/kubernetes/issues/23347) open
tracking the tool to automate the batching procedure.

#### Cherrypicking a doc change

If you are cherrypicking a change which adds a doc, then you also need to run
`build/versionize-docs.sh` in the release branch to versionize that doc.
Ideally, just running `hack/cherry_pick_pull.sh` should be enough, but we are
not there yet: [#18861](https://github.com/kubernetes/kubernetes/issues/18861)

To cherrypick PR 123456 to release-3.14, run the following commands after
running `hack/cherry_pick_pull.sh` and before merging the PR:

```
$ git checkout -b automated-cherry-pick-of-#123456-upstream-release-3.14
origin/automated-cherry-pick-of-#123456-upstream-release-3.14
$ ./build/versionize-docs.sh release-3.14
$ git commit -a -m "Running versionize docs"
$ git push origin automated-cherry-pick-of-#123456-upstream-release-3.14
```

## Cherry Pick Review

Cherry pick pull requests are reviewed differently than normal pull requests. In
particular, they may be self-merged by the release branch owner without fanfare,
in the case the release branch owner knows the cherry pick was already
requested - this should not be the norm, but it may happen.

## Searching for Cherry Picks

See the [cherrypick queue dashboard](http://cherrypick.k8s.io/#/queue) for
status of PRs labeled as `cherrypick-candidate`.

[Contributor License Agreements](http://releases.k8s.io/release-1.4/CONTRIBUTING.md) is
considered implicit for all code within cherry-pick pull requests, ***unless
there is a large conflict***.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/cherry-picks.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
