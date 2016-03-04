<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Pull Request Process
====================

An overview of how pull requests are managed for kubernetes. This document
assumes the reader has already followed the [development guide](development.md)
to set up their environment.

Life of a Pull Request
----------------------

Unless in the last few weeks of a milestone when we need to reduce churn and stabilize, we aim to be always accepting pull requests.

Either the [on call](on-call-rotations.md) manually or the [github "munger"](https://github.com/kubernetes/contrib/tree/master/mungegithub) submit-queue plugin automatically will manage merging PRs.

There are several requirements for the submit-queue to work:
* Author must have signed CLA ("cla: yes" label added to PR)
* No changes can be made since last lgtm label was applied
* k8s-bot must have reported the GCE E2E build and test steps passed (Travis, Jenkins unit/integration, Jenkins e2e)

Additionally, for infrequent or new contributors, we require the on call to apply the "ok-to-merge" label manually.  This is gated by the [whitelist](https://github.com/kubernetes/contrib/blob/master/mungegithub/whitelist.txt).

### Before sending a pull request

The following will save time for both you and your reviewer:

* Enable [pre-commit hooks](development.md#committing-changes-to-your-fork) and verify they pass.
* Verify `hack/verify-generated-docs.sh` passes.
* Verify `hack/test-go.sh` passes.

### Visual overview

![PR workflow](pr_workflow.png)

Other notes
-----------

Pull requests that are purely support questions will be closed and
redirected to [stackoverflow](http://stackoverflow.com/questions/tagged/kubernetes).
We do this to consolidate help/support questions into a single channel,
improve efficiency in responding to requests and make FAQs easier
to find.

Pull requests older than 2 weeks will be closed.  Exceptions can be made
for PRs that have active review comments, or that are awaiting other dependent PRs.
Closed pull requests are easy to recreate, and little work is lost by closing a pull
request that subsequently needs to be reopened. We want to limit the total number of PRs in flight to:
* Maintain a clean project
* Remove old PRs that would be difficult to rebase as the underlying code has changed over time
* Encourage code velocity


Automation
----------

We use a variety of automation to manage pull requests.  This automation is described in detail
[elsewhere.](automation.md)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/pull-requests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
