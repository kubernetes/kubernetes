<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/automation.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Development Automation

## Overview

Kubernetes uses a variety of automated tools in an attempt to relieve developers
of repetitive, low brain power work. This document attempts to describe these
processes.


## Submit Queue

In an effort to
   * reduce load on core developers
   * maintain e2e stability
   * load test github's label feature

We have added an automated [submit-queue]
(https://github.com/kubernetes/contrib/blob/master/mungegithub/mungers/submit-queue.go)
to the
[github "munger"](https://github.com/kubernetes/contrib/tree/master/mungegithub)
for kubernetes.

The submit-queue does the following:

```go
for _, pr := range readyToMergePRs() {
    if testsAreStable() {
        if retestPR(pr) == success {
            mergePR(pr)
        }
    }
}
```

The status of the submit-queue is [online.](http://submit-queue.k8s.io/)

### Ready to merge status

The submit-queue lists what it believes are required on the [merge requirements tab](http://submit-queue.k8s.io/#/info) of the info page. That may be more up to date.

A PR is considered "ready for merging" if it matches the following:
  * The PR must have the label "cla: yes" or "cla: human-approved"
  * The PR must be mergeable. aka cannot need a rebase
  * All of the following github statuses must be green
     * Jenkins GCE Node e2e
     * Jenkins GCE e2e
     * Jenkins unit/integration
  * The PR cannot have any prohibited future milestones (such as a v1.5 milestone during v1.4 code freeze)
  * The PR must have the "lgtm" label. The "lgtm" label is automatically applied
    following a review comment consisting of only "LGTM" (case-insensitive)
  * The PR must not have been updated since the "lgtm" label was applied
  * The PR must not have the "do-not-merge" label

### Merge process

Merges _only_ occur when the [critical builds](http://submit-queue.k8s.io/#/e2e)
are passing. We're open to including more builds here, let us know...

Merges are serialized, so only a single PR is merged at a time, to ensure
against races.

If the PR has the `retest-not-required` label, it is simply merged. If the PR does
not have this label the e2e, unit/integration, and node  tests are re-run. If these
tests pass a second time, the PR will be merged as long as the `critical builds` are
green when this PR finishes retesting.

## Github Munger

We run [github "mungers"](https://github.com/kubernetes/contrib/tree/master/mungegithub).

This runs repeatedly over github pulls and issues and runs modular "mungers"
similar to "mungedocs." The mungers include the 'submit-queue' referenced above along
with numerous other functions. See the README in the link above.

Please feel free to unleash your creativity on this tool, send us new mungers
that you think will help support the Kubernetes development process.

### Closing stale pull-requests

Github Munger will close pull-requests that don't have human activity in the
last 90 days. It will warn about this process 60 days before closing the
pull-request, and warn again 30 days later. One way to prevent this from
happening is to add the "keep-open" label on the pull-request.

Feel free to re-open and maybe add the "keep-open" label if this happens to a
valid pull-request. It may also be a good opportunity to get more attention by
verifying that it is properly assigned and/or mention people that might be
interested. Commenting on the pull-request will also keep it open for another 90
days.

## PR builder

We also run a robotic PR builder that attempts to run tests for each PR.

Before a PR from an unknown user is run, the PR builder bot (`k8s-bot`) asks to
a message from a contributor that a PR is "ok to test", the contributor replies
with that message.  ("please" is optional, but remember to treat your robots with
kindness...)

## FAQ:

#### How can I ask my PR to be tested again for Jenkins failures?

PRs should only need to be manually re-tested if you believe there was a flake
during the original test. All flakes should be filed as an
[issue](https://github.com/kubernetes/kubernetes/issues?q=is%3Aopen+is%3Aissue+label%3Akind%2Fflake).
Once you find or file a flake a contributer (this may be you!) should request
a retest with "@k8s-bot test this issue: #NNNNN", where NNNNN is replaced with
the issue number you found or filed.

Any pushes of new code to the PR will automatically trigger a new test. No human
interraction is required.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/automation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
