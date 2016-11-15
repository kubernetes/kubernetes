## Kubernetes "Github and Build-cop" Rotation

### Preqrequisites

* Ensure you have [write access to http://github.com/kubernetes/kubernetes](https://github.com/orgs/kubernetes/teams/kubernetes-maintainers)
  * Test your admin access by e.g. adding a label to an issue.

### Traffic sources and responsibilities

* GitHub Kubernetes [issues](https://github.com/kubernetes/kubernetes/issues)
and [pulls](https://github.com/kubernetes/kubernetes/pulls): Your job is to be
the first responder to all new issues and PRs. If you are not equipped to do
this (which is fine!), it is your job to seek guidance!

  * Support issues should be closed and redirected to Stackoverflow (see example
response below).

  * All incoming issues should be tagged with a team label
(team/{api,ux,control-plane,node,cluster,csi,redhat,mesosphere,gke,release-infra,test-infra,none});
for issues that overlap teams, you can use multiple team labels

    * There is a related concept of "Github teams" which allow you to @ mention
a set of people; feel free to @ mention a Github team if you wish, but this is
not a substitute for adding a team/* label, which is required

      * [Google teams](https://github.com/orgs/kubernetes/teams?utf8=%E2%9C%93&query=goog-)
      * [Redhat teams](https://github.com/orgs/kubernetes/teams?utf8=%E2%9C%93&query=rh-)
      * [SIGs](https://github.com/orgs/kubernetes/teams?utf8=%E2%9C%93&query=sig-)

    * If the issue is reporting broken builds, broken e2e tests, or other
obvious P0 issues, label the issue with priority/P0 and assign it to someone.
This is the only situation in which you should add a priority/* label
      * non-P0 issues do not need a reviewer assigned initially

    * Assign any issues related to Vagrant to @derekwaynecarr (and @mention him
in the issue)

  * All incoming PRs should be assigned a reviewer.

    * unless it is a WIP (Work in Progress), RFC (Request for Comments), or design proposal.
    * An auto-assigner [should do this for you] (https://github.com/kubernetes/kubernetes/pull/12365/files)
    * When in doubt, choose a TL or team maintainer of the most relevant team; they can delegate

  * Keep in mind that you can @ mention people in an issue/PR to bring it to
their attention without assigning it to them. You can also @ mention github
teams, such as @kubernetes/goog-ux or @kubernetes/kubectl

  * If you need help triaging an issue or PR, consult with (or assign it to)
@brendandburns, @thockin, @bgrant0607, @quinton-hoole, @davidopp, @dchen1107,
@lavalamp (all U.S. Pacific Time) or @fgrzadkowski (Central European Time).

  * At the beginning of your shift, please add team/* labels to any issues that
have fallen through the cracks and don't have one. Likewise, be fair to the next
person in rotation: try to ensure that every issue that gets filed while you are
on duty is handled. The Github query to find issues with no team/* label is:
[here](https://github.com/kubernetes/kubernetes/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+-label%3Ateam%2Fcontrol-plane+-label%3Ateam%2Fmesosphere+-label%3Ateam%2Fredhat+-label%3Ateam%2Frelease-infra+-label%3Ateam%2Fnone+-label%3Ateam%2Fnode+-label%3Ateam%2Fcluster+-label%3Ateam%2Fux+-label%3Ateam%2Fapi+-label%3Ateam%2Ftest-infra+-label%3Ateam%2Fgke+-label%3A"team%2FCSI-API+Machinery+SIG"+-label%3Ateam%2Fhuawei+-label%3Ateam%2Fsig-aws).

Example response for support issues:

```code
Please re-post your question to [stackoverflow]
(http://stackoverflow.com/questions/tagged/kubernetes).

We are trying to consolidate the channels to which questions for help/support
are posted so that we can improve our efficiency in responding to your requests,
and to make it easier for you to find answers to frequently asked questions and
how to address common use cases.

We regularly see messages posted in multiple forums, with the full response
thread only in one place or, worse, spread across multiple forums. Also, the
large volume of support issues on github is making it difficult for us to use
issues to identify real bugs.

The Kubernetes team scans stackoverflow on a regular basis, and will try to
ensure your questions don't go unanswered.

Before posting a new question, please search stackoverflow for answers to
similar questions, and also familiarize yourself with:

  * [user guide](http://kubernetes.io/docs/user-guide/)
  * [troubleshooting guide](http://kubernetes.io/docs/admin/cluster-troubleshooting/)

Again, thanks for using Kubernetes.

The Kubernetes Team
```

### Build-copping

* The [merge-bot submit queue](http://submit-queue.k8s.io/)
([source](https://github.com/kubernetes/contrib/tree/master/mungegithub/mungers/submit-queue.go))
should auto-merge all eligible PRs for you once they've passed all the relevant
checks mentioned below and all [critical e2e tests]
(https://goto.google.com/k8s-test/view/Critical%20Builds/) are passing. If the
merge-bot been disabled for some reason, or tests are failing, you might need to
do some manual merging to get things back on track.

* Once a day or so, look at the [flaky test builds]
(https://goto.google.com/k8s-test/view/Flaky/); if they are timing out, clusters
are failing to start, or tests are consistently failing (instead of just
flaking), file an issue to get things back on track.

* Jobs that are not in [critical e2e tests](https://goto.google.com/k8s-test/view/Critical%20Builds/)
or [flaky test builds](https://goto.google.com/k8s-test/view/Flaky/) are not
your responsibility to monitor. The `Test owner:` in the job description will be
automatically emailed if the job is failing.

* If you are oncall, ensure that PRs confirming to the following
pre-requisites are being merged at a reasonable rate:

  * [Have been LGTMd](https://github.com/kubernetes/kubernetes/labels/lgtm)
  * Pass Travis and Jenkins per-PR tests.
  * Author has signed CLA if applicable.


* Although the shift schedule shows you as being scheduled Monday to Monday,
  working on the weekend is neither expected nor encouraged.  Enjoy your time
  off.

* When the build is broken, roll back the PRs responsible ASAP

* When E2E tests are unstable, a "merge freeze" may be instituted. During a
merge freeze:

  * Oncall should slowly merge LGTMd changes throughout the day while monitoring
E2E to ensure stability.

  * Ideally the E2E run should be green, but some tests are flaky and can fail
randomly (not as a result of a particular change).
      * If a large number of tests fail, or tests that normally pass fail, that
is an indication that one or more of the PR(s) in that build might be
problematic (and should be reverted).
      * Use the Test Results Analyzer to see individual test history over time.


* Flake mitigation

  * Tests that flake (fail a small percentage of the time) need an issue filed
against them. Please read [this](flaky-tests.md#filing-issues-for-flaky-tests);
the build cop is expected to file issues for any flaky tests they encounter.

  * It's reasonable to manually merge PRs that fix a flake or otherwise mitigate it.

### Contact information

[@k8s-oncall](https://github.com/k8s-oncall) will reach the current person on
call.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/on-call-build-cop.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
