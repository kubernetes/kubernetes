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
[here](http://releases.k8s.io/release-1.3/docs/devel/on-call-user-support.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Kubernetes "User Support" Rotation

### Traffic sources and responsibilities

* [StackOverflow](http://stackoverflow.com/questions/tagged/kubernetes) and
[ServerFault](http://serverfault.com/questions/tagged/google-kubernetes):
Respond to any thread that has no responses and is more than 6 hours old (over
time we will lengthen this timeout to allow community responses). If you are not
equipped to respond, it is your job to redirect to someone who can.

  * [Query for unanswered Kubernetes StackOverflow questions](http://stackoverflow.com/search?q=%5Bkubernetes%5D+answers%3A0)
  * [Query for unanswered Kubernetes ServerFault questions](http://serverfault.com/questions/tagged/google-kubernetes?sort=unanswered&pageSize=15)
  * Direct poorly formulated questions to [stackoverflow's tips about how to ask](http://stackoverflow.com/help/how-to-ask)
  * Direct off-topic questions to [stackoverflow's policy](http://stackoverflow.com/help/on-topic)

* [Slack](https://kubernetes.slack.com) ([registration](http://slack.k8s.io)):
Your job is to be on Slack, watching for questions and answering or redirecting
as needed. Also check out the [Slack Archive](http://kubernetes.slackarchive.io/).

* [Email/Groups](https://groups.google.com/forum/#!forum/google-containers):
Respond to any thread that has no responses and is more than 6 hours old (over
time we will lengthen this timeout to allow community responses). If you are not
equipped to respond, it is your job to redirect to someone who can.

* [Legacy] [IRC](irc://irc.freenode.net/#google-containers)
(irc.freenode.net #google-containers): watch IRC for questions and try to
redirect users to Slack. Also check out the
[IRC logs](https://botbot.me/freenode/google-containers/).

In general, try to direct support questions to:

1. Documentation, such as the [user guide](../user-guide/README.md) and
[troubleshooting guide](../troubleshooting.md)

2. Stackoverflow

If you see questions on a forum other than Stackoverflow, try to redirect them
to Stackoverflow. Example response:

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
  * [troubleshooting guide](http://kubernetes.io/docs/troubleshooting/)

Again, thanks for using Kubernetes.

The Kubernetes Team
```

If you answer a question (in any of the above forums) that you think might be
useful for someone else in the future, *please add it to one of the FAQs in the
wiki*:

* [User FAQ](https://github.com/kubernetes/kubernetes/wiki/User-FAQ)
* [Developer FAQ](https://github.com/kubernetes/kubernetes/wiki/Developer-FAQ)
* [Debugging FAQ](https://github.com/kubernetes/kubernetes/wiki/Debugging-FAQ).

Getting it into the FAQ is more important than polish. Please indicate the date
it was added, so people can judge the likelihood that it is out-of-date (and
please correct any FAQ entries that you see contain out-of-date information).

### Contact information

[@k8s-support-oncall](https://github.com/k8s-support-oncall) will reach the
current person on call.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/on-call-user-support.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
