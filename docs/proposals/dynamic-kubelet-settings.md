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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for improving Kubelet config by versioning it and making it manageable via the API server.

## Motivation

The Kubelet is currently configured via command-line flags. This is painful for a number of reasons:
- It makes it difficult to change the way Kubelets are configured in a running cluster, because it is often tedious to change the Kubelet startup configuration.
- It makes it difficult to manage different Kubelet configurations for different nodes, e.g. if you want to canary a new config.
- The current lack of versioned Kubelet configuration means that any time we change Kubelet flags, we risk breaking someone's setup.
- Having the ability to pass config in the same format on-disk as via the API server will make it easier to bootstrap master nodes before the API server is available.


## Goals of this design:

- Add a well-defined, versioned configmap object for Kubelet configuration.
- Add/Remove/Update global and per-node Kubelet config on the fly via the API server.
- Provide a reliable way to get the configuration of a given Kubelet along with its source (remote or on disk or flags).
- We want the Kubelet to work with the built-in defaults on all distros we test against.
- Work towards deprecating flags in favor of on-disk config and config from the API server. Master nodes will be bootstrapped from on-disk config.
- Make it possible to opt-out of remote configuration e.g. to protect the master from bad config updates.

## Design

### Definitions

- `kubelet-default`: The built-in default config that every Kubelet starts up with.
- `api-default`: The defaults provided for omitted/nil fields by the API server.
- `on-disk-config`: `KubeletConfiguration` provided on disk, used as a fallback when the Kubelet cannot contact the API server.
- `minimal-config`: Conceptually, what the Kubelet needs to start up and contact the API server. This probably consists of the location of the API server and credentials to access the API server, and might be provided via flags or via `on-disk-config`.
- `global-kubelet-configuration`: the `KubeletConfiguration` configmap set via the API server which is shared by all Kubelets in your cluster except those which have a `<node-name>-kubelet-configuration`.
- `<node-name>-kubelet-configuration`: the `KubeletConfiguration` configmap specific to the node named `<node-name>`, also set via the API server.

### Well-defined configmap type for Kubelet configuration

- There is an alpha version of the `KubeletConfiguration` type: external in `pkg/componentconfig/v1alpha1/types.go`, internal in `pkg/componentconfig/types.go`.
- We likely want to group related fields of the `KubeletConfiguration` into substructures. Please note that `v1alpha1` will not be regrouped, but eventually we will start working on this.

@timstclair and I did some brainstorming and came up with some rough potential categories:

---
- Networking
- Runtimes (only one of these, in addition to Common, should be specified for a given Kubelet)
    + Common
    + Rkt
    + Docker
- Events
- Security/Policies
- Images (might be better as a runtime sub-category)
- Resource Management
    + GC/Eviction
    + Allocatable/SystemReserved/KubeReserved
    + Custom resources? e.g. NvidiaGPUs?
- Cloud-provider-specific settings
- Cluster/API
    + KubeAPIQPS
    + ContentType
- SystemSpecification (this might be a non-category, all of the available hardware should theoretically be detectable by the Kubelet)
    + NodeIP (probably belongs in networking)
    + NvidiaGPUs could be here, but shouldn't the Kubelet be able to detect this?
- Experimental/Volatile Features (could be tricky, might be better off using alpha versioning to separate experiments from the stable API)

---

### Bringing up a master node

- Start the Kubelet with `minimal-config`
- On-disk static pod manifests are used to bring up the API server, scheduler, controller-manager, etc.
Once the master is ready, the user creates some configs via the API:
- They might create a `global-kubelet-configuration`
- They might create a `<node-name>-kubelet-configuration`, e.g. if they want a master-specific configuration.
- Then the Kubelets' configuration sync loops will see the new config, and the Kubelets will restart to take up that new config.

### Bringing up a Kubelet

- Start the Kubelet with `minimal-config`
- If `<current-node-name>-kubelet-config` exists on the API server, download and use the `<current-node-name>-kubelet-config`
- else if `global-kubelet-config` exists on the API server, download and use the `global-kubelet-config`.
- else if e.g. an `on-disk-config` flag is set, defer to the node's on-disk config.
- else defer to config from flags
- Finally, start the configuration sync loop in a background thread


### Storing config locally (checkpointing)

Initially, the Kubelet will store the most recent configmap it got from the API server internally in memory, and compare that to new values it gets from the API server. It will restart when these configmaps differ in order to apply the new config.

This is a rudimentary solution, and @aaronlevy brought up some good reasons to checkpoint the config, and store it externally from the Kubelet:
>Without being privy to prior discussions, I might argue for saving the config state outside the kubelet (and reboot simply picks up this local / on-disk config).

>A reason for this would be to minimize the delta of `minimal-config` + `kubelet-default` to the api-defined config through lifecycle of the kubelet. A specific case that has been brought up in the past are cgroup related settings (e.g. kubelet + minimal-config starts static pods, then gets a new config with different cgroup settings, then starts more pods). Or maybe that's just covered by a "don't do that" clause.

I think it might be worth opening a separate issue to discuss a general solution for checkpointing config (unless one already exists, but I haven't seen one). There is https://github.com/kubernetes/kubernetes/issues/489, but that seems more geared towards checkpointing pod manifests. Is that, in fact, the right place to discuss this, or should I open a separate issue (if I open a separate issue, I'll post a link here)?




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/dynamic-kubelet-settings.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
