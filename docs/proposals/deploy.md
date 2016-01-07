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
[here](http://releases.k8s.io/release-1.1/docs/proposals/deploy.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Deploy through CLI](#deploy-through-cli)
  - [Motivation](#motivation)
  - [Requirements](#requirements)
  - [Related `kubectl` Commands](#related-kubectl-commands)
    - [`kubectl run`](#kubectl-run)
    - [`kubectl scale` and `kubectl autoscale`](#kubectl-scale-and-kubectl-autoscale)
    - [`kubectl rollout`](#kubectl-rollout)
    - [`kubectl set`](#kubectl-set)
    - [Mutating Operations](#mutating-operations)
    - [Example](#example)
  - [Support in Deployment](#support-in-deployment)
    - [Deployment Status](#deployment-status)
    - [Deployment Version](#deployment-version)
    - [Pause Deployments](#pause-deployments)
    - [Perm-failed Deployments](#perm-failed-deployments)

<!-- END MUNGE: GENERATED_TOC -->

# Deploy through CLI

## Motivation

Users can use Deployments or `kubectl rolling-update` to deploy in their Kubernetes clusters. A Deployment provides declarative update for Pods and ReplicationControllers, whereas `rolling-update` allows the users to update their earlier deployment without worrying about schemas and configurations. Users need a way that's similar to `rolling-update` to manage their Deployments more easily.

`rolling-update` expects ReplicationController as the only resource type it deals with. It's not trivial to support exactly the same behavior with Deployment, which requires:
- Print out scaling up/down events.
- Stop the deployment if users press Ctrl-c.
- The controller should not make any more changes once the process ends. (Delete the deployment when status.replicas=status.updatedReplicas=spec.replicas)

So, instead, this document proposes another way to support easier deployment management via Kubernetes CLI (`kubectl`).

## Requirements

The followings are operations we need to support for the users to easily managing deployments:

- **Create**: To create deployments.
- **Rollback**: To restore to an earlier version of deployment.
- **Watch the status**: To watch for the status update of deployments.
- **Pause/resume**: To pause a deployment mid-way, and to resume it. (A use case is to support canary deployment.)
- **Version information**: To record and show version information that's meaningful to users. This can be useful for rollback.

## Related `kubectl` Commands

### `kubectl run`

`kubectl run` should support the creation of Deployment (already implemented) and DaemonSet resources.

### `kubectl scale` and `kubectl autoscale`

Users may use `kubectl scale` or `kubectl autoscale` to scale up and down Deployments (both already implemented).

### `kubectl rollout`

`kubectl rollout` supports both Deployment and DaemonSet. It has the following subcommands:
- `kubectl rollout undo` works like rollback; it allows the users to rollback to a previous version of deployment.
- `kubectl rollout pause` allows the users to pause a deployment.
- `kubectl rollout resume` allows the users to resume a paused deployment.
- `kubectl rollout status` shows the status of a deployment.
- `kubectl rollout history` shows meaningful version information of all previous deployments.

### `kubectl set`

`kubectl set` has the following subcommands:
- `kubectl set env` allows the users to set environment variables of Kubernetes resources. It should support any object that contains a single, primary PodTemplate (such as Pod, ReplicationController, ReplicaSet, Deployment, and DaemonSet).
- `kubectl set image` allows the users to update multiple images of Kubernetes resources. Users will use `--container` and `--image` flags to update the image of a container. It should support anything that has a PodTemplate.

`kubectl set` should be used for things that are common and commonly modified. Other possible future commands include:
- `kubectl set volume`
- `kubectl set limits`
- `kubectl set security`
- `kubectl set port`

### Mutating Operations

Other means of mutating Deployments and DaemonSets, including `kubectl apply`, `kubectl edit`, `kubectl replace`, `kubectl patch`, `kubectl label`, and `kubectl annotate`, may trigger rollouts if they modify the pod template.

`kubectl create` and `kubectl delete`, for creating and deleting Deployments and DaemonSets, are also relevant.

### Example

With the commands introduced above, here's an example of deployment management:

```console
# Create a Deployment
$ kubectl run nginx --image=nginx --replicas=2 --generator=deployment/v1beta1

# Watch the Deployment status
$ kubectl rollout status deployment/nginx

# Update the Deployment 
$ kubectl set image deployment/nginx --container=nginx --image=nginx:<some-version>

# Pause the Deployment
$ kubectl rollout pause deployment/nginx

# Resume the Deployment
$ kubectl rollout resume deployment/nginx

# Check the change history (deployment versions)
$ kubectl rollout history deployment/nginx

# Rollback to a previous version.
$ kubectl rollout undo deployment/nginx --to-version=<version>
```

## Support in Deployment

### Deployment Status

Deployment status should summarize information about Pods, which includes:
- The number of pods of each version.
- The number of ready/not ready pods.

See issue [#17164](https://github.com/kubernetes/kubernetes/issues/17164).

### Deployment Version

We store previous deployment versions information in deployment annotation `kubectl.kubernetes.io/deployment-version-<time>`. The value stored in the annotation can be a spec hash, a strategic merge patch or the kubectl commands previously executed. We choose strategic merge patch since it's more human-readable than spec hash, and commands like `kubectl edit` won't provide enough useful information.

### Pause Deployments

Users sometimes need to temporarily disable a deployment. See issue [#14516](https://github.com/kubernetes/kubernetes/issues/14516).

### Perm-failed Deployments

The deployment could be marked as "permanently failed" for a given spec hash so that the system won't continue thrashing on a doomed deployment. See issue [#14519](https://github.com/kubernetes/kubernetes/issues/14519).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/deploy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
