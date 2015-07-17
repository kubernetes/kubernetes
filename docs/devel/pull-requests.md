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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/pull-requests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Pull Request Process
====================

An overview of how we will manage old or out-of-date pull requests.

Process
-------

We will close any pull requests older than two weeks.

Exceptions can be made for PRs that have active review comments, or that are awaiting other dependent PRs.  Closed pull requests are easy to recreate, and little work is lost by closing a pull request that subsequently needs to be reopened.

We want to limit the total number of PRs in flight to:
* Maintain a clean project
* Remove old PRs that would be difficult to rebase as the underlying code has changed over time
* Encourage code velocity

RC to v1.0 Pull Requests
------------------------

Between the first RC build (~6/22) and v1.0, we will adopt a higher bar for PR merges.  For v1.0 to be a stable release, we need to ensure that any fixes going in are very well tested and have a low risk of breaking anything.  Refactors and complex changes will be rejected in favor of more strategic and smaller workarounds.

These PRs require:
* A risk assessment by the code author in the PR.  This should outline which parts of the code are being touched, the risk of regression, and complexity of the code.
* Two LGTMs from experienced reviewers.

Once those requirements are met, they will be labeled [ok-to-merge](https://github.com/GoogleCloudPlatform/kubernetes/pulls?utf8=%E2%9C%93&q=is%3Aopen+is%3Apr+label%3Aok-to-merge) and can be merged.

These restrictions will be relaxed after v1.0 is released.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/pull-requests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
