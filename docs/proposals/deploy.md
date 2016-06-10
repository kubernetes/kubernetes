<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

Users can use [Deployments](../user-guide/deployments.md) or [`kubectl rolling-update`](../user-guide/kubectl/kubectl_rolling-update.md) to deploy in their Kubernetes clusters. A Deployment provides declarative update for Pods and ReplicationControllers, whereas `rolling-update` allows the users to update their earlier deployment without worrying about schemas and configurations. Users need a way that's similar to `rolling-update` to manage their Deployments more easily.

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
- `kubectl rollout pause` allows the users to pause a deployment. See [pause deployments](#pause-deployments).
- `kubectl rollout resume` allows the users to resume a paused deployment.
- `kubectl rollout status` shows the status of a deployment.
- `kubectl rollout history` shows meaningful version information of all previous deployments. See [development version](#deployment-version).
- `kubectl rollout retry` retries a failed deployment. See [perm-failed deployments](#perm-failed-deployments).

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

We store previous deployment version information in annotations `rollout.kubectl.kubernetes.io/change-source` and `rollout.kubectl.kubernetes.io/version` of replication controllers of the deployment, to support rolling back changes as well as for the users to view previous changes with `kubectl rollout history`.
- `rollout.kubectl.kubernetes.io/change-source`, which is optional, records the kubectl command of the last mutation made to this rollout. Users may use `--record` in `kubectl` to record current command in this annotation.
- `rollout.kubectl.kubernetes.io/version` records a version number to distinguish the change sequence of a deployment's
replication controllers. A deployment obtains the largest version number from its replication controllers and increments the number by 1 upon update or creation of the deployment, and update the version annotation of its new replication controller.

When the users perform a rollback, i.e. `kubectl rollout undo`, the deployment first looks at its existing replication controllers, regardless of their number of replicas. Then it finds the one with annotation `rollout.kubectl.kubernetes.io/version` that either contains the specified rollback version number or contains the second largest version number among all the replication controllers (current new replication controller should obtain the largest version number) if the user didn't specify any version number (the user wants to rollback to the last change). Lastly, it
starts scaling up that replication controller it's rolling back to, and scaling down the current ones, and then update the version counter and the rollout annotations accordingly.

Note that a deployment's replication controllers use PodTemplate hashes (i.e. the hash of `.spec.template`) to distinguish from each others. When doing rollout or rollback, a deployment reuses existing replication controller if it has the same PodTemplate, and its `rollout.kubectl.kubernetes.io/change-source` and `rollout.kubectl.kubernetes.io/version` annotations will be updated by the new rollout. At this point, the earlier state of this replication controller is lost in history. For example, if we had 3 replication controllers in
deployment history, and then we do a rollout with the same PodTemplate as version 1, then version 1 is lost and becomes version 4 after the rollout.

To make deployment versions more meaningful and readable for the users, we can add more annotations in the future. For example, we can add the following flags to `kubectl` for the users to describe and record their current rollout:
- `--description`: adds `description` annotation to an object when it's created to describe the object.
- `--note`: adds `note` annotation to an object when it's updated to record the change.
- `--commit`: adds `commit` annotation to an object with the commit id.

### Pause Deployments

Users sometimes need to temporarily disable a deployment. See issue [#14516](https://github.com/kubernetes/kubernetes/issues/14516).

### Perm-failed Deployments

The deployment could be marked as "permanently failed" for a given spec hash so that the system won't continue thrashing on a doomed deployment. The users can retry a failed deployment with `kubectl rollout retry`. See issue [#14519](https://github.com/kubernetes/kubernetes/issues/14519).



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/deploy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
