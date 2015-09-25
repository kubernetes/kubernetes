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
[here](http://releases.k8s.io/release-1.0/docs/roadmap.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Roadmap

## Kubernetes 1.1

### Timeline

We are targetting late October for our 1.1 release of Kubernetes.  We plan on cutting a first release candidate
in early October.  We will enter feature freeze for the 1.1 release on September 21st.  Note this does not mean
that the master branch is fully frozen, but all 1.1 features *must* be in by September 21st and large-scale
refactors of the codebase will be blocked until the 1.1 release is finalized to ensure easy cherry-picks.

### Scope

The 1.1 release of Kubernetes will be a purely additive releases, the `v1` API will be maintained, with a set
of newly added features.

#### Blocking Features

The following features are considered blocking for the 1.1 release:
   * Docker 1.8.x
   * Graceful pod termination
   * IPtables based kube-proxy (tbd if this is the default for all platforms)
   * Improvements to kubectl usability and features
   * Support for 250 node clusters
   * Horizontal Pod autoscaling
   * Support for experimental APIs and API groups.
   * Job objects

#### Nice to have features

The following features will be part of 1.1 if complete, but will not block the release:
   * Deployment API
   * ScheduledJob API
   * Daemon Controller
   * ConfigData API
   * HTTP(S) load balancer support
   * Rolling update improvements
   * Third party CRUD resources

## Post 1.1

We're in the process of prioritizing changes to be made after 1.1.

Please watch the [Github milestones] (https://github.com/kubernetes/kubernetes/milestones) for our future plans.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/roadmap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
