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
[here](http://releases.k8s.io/release-1.3/docs/devel/on-call-rotations.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Kubernetes On-Call Rotations

### Kubernetes "first responder" rotations

Kubernetes has generated a lot of public traffic: email, pull-requests, bugs,
etc. So much traffic that it's becoming impossible to keep up with it all! This
is a fantastic problem to have. In order to be sure that SOMEONE, but not
EVERYONE on the team is paying attention to public traffic, we have instituted
two "first responder" rotations, listed below. Please read this page before
proceeding to the pages linked below, which are specific to each rotation.

Please also read our [notes on OSS collaboration](collab.md), particularly the
bits about hours. Specifically, each rotation is expected to be active primarily
during work hours, less so off hours.

During regular workday work hours of your shift, your primary responsibility is
to monitor the traffic sources specific to your rotation. You can check traffic
in the evenings if you feel so inclined, but it is not expected to be as highly
focused as work hours. For weekends, you should check traffic very occasionally
(e.g. once or twice a day). Again, it is not expected to be as highly focused as
workdays. It is assumed that over time, everyone will get weekday and weekend
shifts, so the workload will balance out.

If you can not serve your shift, and you know this ahead of time, it is your
responsibility to find someone to cover and to change the rotation. If you have
an emergency, your responsibilities fall on the primary of the other rotation,
who acts as your secondary. If you need help to cover all of the tasks, partners
with oncall rotations (e.g.,
[Redhat](https://github.com/orgs/kubernetes/teams/rh-oncall)).

If you are not on duty you DO NOT need to do these things. You are free to focus
on "real work".

Note that Kubernetes will occasionally enter code slush/freeze, prior to
milestones. When it does, there might be changes in the instructions (assigning
milestones, for instance).

* [Github and Build Cop Rotation](on-call-build-cop.md)
* [User Support Rotation](on-call-user-support.md)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/on-call-rotations.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
