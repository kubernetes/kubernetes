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
[here](http://releases.k8s.io/release-1.4/docs/proposals/kubelet-checkpointing.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubelet checkpointing

## Background

* [Kubelet checkpointing or something](https://github.com/kubernetes/kubernetes/issues/489)
* [self-hosting](https://github.com/kubernetes/kubernetes/issues/246)
* [Kubelet as a Container and Self-Hosted Kubernetes](https://docs.google.com/document/d/1_I6xT0XHCoOqZUT-dtpxzwvYpTR5JmFQY0S4gL2PPkU/edit#)
* [user-space checkpointing](https://github.com/kubernetes-incubator/bootkube/tree/master/cmd/checkpoint)
* [sig-cluster-lifecycle discussion on checkpointing](https://docs.google.com/document/d/172T6T9R35wbs5wYERnne-2-Pivy4wL58PM5mVbEbkHo/edit#)
* [kubelet persisten store](https://github.com/kubernetes/kubernetes/pull/13861)

## Abstract

Introduce a mechanism for checkpointing pods, configmaps and secrets.

## Motivation

Self-hosting brings a lot of advantages, but the reliance of the kubelet on the apiserver being available (and thus, etcd, routing daemons and the like, if they're also self-hosted) means that recovering from such a failure is hard.

This proposal aims to support bringing a self-hosted cluster back from a cold-boot scenario.
i.e. All nodes are down simultaneously (Say, massive power failure in a bare metal cluster)

## Main proposal

Kubelet will maintain a persistent store on disk, containing:
* all pod manifests scheduled to the node
* all configmaps referenced by the above
* all secrets referenced by the above

When a pod addition, change, or delete is noticed, it's persisted first and then acted upon.

During startup, if the database file does not exist:
* The database file is created
* Kubelet waits for the apiserver in `allSourcesReady` as normal
If the database file does exist:
* The internal store is populated from the persisted manifests
* The apiserver source is marked as ready (though obviously it may not have been contacted yet)
* kubelet proceeds to create the pods, when the apiserver is reachable things proceed as normal, as if kubelet had been connected to the apiserver all along.

### Implementation details

* Add `Checkpointing=true|false (ALPHA - default=false)` to `--feature-gates`
* Store file will be at <root_dir>/checkpoint-store.db
* Secrets will be encrypted on disk using the kubelet's private key.
* * This preserves the current assumption that secrets are not accessible on a node when it's powered off
* * If this key was compromised the kubelet could just ask the apiserver for the keys anyway, so we're in the same situation
* * In particular, if the deployment has been set up with a secure etcd, secure provisioning of kubelet keys etc, this will not then introduce additional unencrypted copies of the secrets.
* There will be an annotation such that a pod can opt out of checkpointing

### Open questions
* What will the store be on disk?
* What's the opt-out annotation?


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-checkpointing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
