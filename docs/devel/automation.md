<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Development Automation

## Overview

Kubernetes uses a variety of automated tools in an attempt to relieve developers of repeptitive, low
brain power work.  This document attempts to describe these processes.


## Submit Queue

In an effort to
   * reduce load on core developers
   * maintain e2e stability
   * load test githubs label feature

We have added an automated [submit-queue](https://github.com/kubernetes/contrib/tree/master/submit-queue)
for kubernetes.

The submit-queue does the following:

```go
for _, pr := range readyToMergePRs() {
    if testsAreStable() {
        mergePR(pr)
    }
}
```

The status of the submit-queue is [online.](http://submit-queue.k8s.io/)

### Ready to merge status

A PR is considered "ready for merging" if it matches the following:
   * it has the `lgtm` label, and that `lgtm` is newer than the latest commit
   * it has passed the cla pre-submit and has the `cla:yes` label
   * it has passed the travis and shippable pre-submit tests
   * one (or all) of
      * its author is in kubernetes/contrib/submit-queue/whitelist.txt
      * its author is in contributors.txt via the github API.
      * the PR has the `ok-to-merge` label
   * One (or both of)
      * it has passed the Jenkins e2e test
      * it has the `e2e-not-required` label

Note that the combined whitelist/committer list is available at [submit-queue.k8s.io](http://submit-queue.k8s.io)

### Merge process

Merges _only_ occur when the `critical builds` (Jenkins e2e for gce, gke, scalability, upgrade) are passing.
We're open to including more builds here, let us know...

Merges are serialized, so only a single PR is merged at a time, to ensure against races.

If the PR has the `e2e-not-required` label, it is simply merged.
If the PR does not have this label, e2e tests are re-run, if these new tests pass, the PR is merged.

If e2e flakes or is currently buggy, the PR will not be merged, but it will be re-run on the following
pass.

## Github Munger

We also run a [github "munger"](https://github.com/kubernetes/contrib/tree/master/mungegithub)

This runs repeatedly over github pulls and issues and runs modular "mungers" similar to "mungedocs"

Currently this runs:
   * blunderbuss - Tries to automatically find an owner for a PR without an owner, uses mapping file here:
        https://github.com/kubernetes/contrib/blob/master/mungegithub/blunderbuss.yml
   * needs-rebase - Adds `needs-rebase` to PRs that aren't currently mergeable, and removes it from those that are.
   * size - Adds `size/xs` - `size/xxl` labels to PRs
   * ok-to-test - Adds the `ok-to-test` message to PRs that have an `lgtm` but the e2e-builder would otherwise not test due to whitelist
   * ping-ci - Attempts to ping the ci systems (Travis/Shippable) if they are missing from a PR.
   * lgtm-after-commit - Removes the `lgtm` label from PRs where there are commits that are newer than the `lgtm` label

In the works:
   * issue-detector - machine learning for determining if an issue that has been filed is a `support` issue, `bug` or `feature`

Please feel free to unleash your creativity on this tool, send us new mungers that you think will help support the Kubernetes development process.

## PR builder

We also run a robotic PR builder that attempts to run e2e tests for each PR.

Before a PR from an unknown user is run, the PR builder bot (`k8s-bot`) asks to a message from a
contributor that a PR is "ok to test", the contributor replies with that message.  Contributors can also
add users to the whitelist by replying with the message "add to whitelist" ("please" is optional, but
remember to treat your robots with kindness...)

If a PR is approved for testing, and tests either haven't run, or need to be re-run, you can ask the
PR builder to re-run the tests.  To do this, reply to the PR with a message that begins with `@k8s-bot test this`, this should trigger a re-build/re-test.


## FAQ:

#### How can I ask my PR to be tested again for Jenkins failures?

Right now you have to ask a contributor (this may be you!) to re-run the test with "@k8s-bot test this"

### How can I kick Shippable to re-test on a failure?

Right now the easiest way is to close and then immediately re-open the PR.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/automation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
