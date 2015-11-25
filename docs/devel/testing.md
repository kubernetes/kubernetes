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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/devel/testing.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes Commit Queue Testing
===============================

A quick overview of how we add, remove and recycle tests from CI.

## What is CI?

Throughout this document we will refer to CI as any suite of e2e tests that can potentially hold up the submit queue, this means Kubernetes PRs must pass these tests prior to getting merged.

## Adding a test to CI

When first adding a test it should *not* go straight into CI, because failures block ordinary development. A test should only be added to CI after is has been running in some non-CI suite long enought to establish a track record showing that the test does not fail when run against *working* software. A suite named `flaky` exists, and can be overloaded to mean `experimental` and used for this reason (can it really?). In addition to this track record, consider the following as requirements:
* The test must be short (20m?)
* Failures must indicate that the product is unfit (TODO: establish a firmer bar, what I'm trying to say is testing random controller X in CI doesn't help anyone, but maybe it does if controller X is a cluster addon, or our largest customer wants X, or what?)
* Failures must reliably indicate a bug in the product, not a bug in the test

(TODO: is there a parallelism requirement here?)

## Moving a test out of CI

Do *not* move a test to flaky as soon as it starts failing just to clear up the submit queue, this risks introducing more bugs and compounding the problem even further (TODO: or do this? but why). Build cop can use their better judgement to call a test `flaky`. This means it fails for presumably random reasons, once in X runs (TODO: is X == 0?). Move flaky tests out of CI, create a P0/1 bug and try to triage it along to the right person. Adding the `kind/flake` label on github will grab the attention of the grumpy CI shamer bot, which will include the bug in its daily report.

If your test got moved to flaky, it must demonstrate the run of greens required for getting added to CI once again (or nah?).

## Non CI channels for testing

If you want to test against Kubernetes but your test doesn't meet the following requirements, peruse [this list](../../hack/jenkins/e2e.sh) and add your test in whatever makes sense (TODO: What about release-lists, shouldn't they mirror other lists?).




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
