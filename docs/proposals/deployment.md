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
[here](http://releases.k8s.io/release-1.3/docs/proposals/deployment.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Deployment

## Abstract

A proposal for implementing a new resource - Deployment - which will enable
declarative config updates for Pods and ReplicationControllers.

Users will be able to create a Deployment, which will spin up
a ReplicationController to bring up the desired pods.
Users can also target the Deployment at existing ReplicationControllers, in
which case the new RC will replace the existing ones. The exact mechanics of
replacement depends on the DeploymentStrategy chosen by the user.
DeploymentStrategies are explained in detail in a later section.

## Implementation

### API Object

The `Deployment` API object will have the following structure:

```go
type Deployment struct {
  TypeMeta
  ObjectMeta

  // Specification of the desired behavior of the Deployment.
  Spec DeploymentSpec

  // Most recently observed status of the Deployment.
  Status DeploymentStatus
}

type DeploymentSpec struct {
  // Number of desired pods. This is a pointer to distinguish between explicit
  // zero and not specified. Defaults to 1.
  Replicas *int

  // Label selector for pods. Existing ReplicationControllers whose pods are
  // selected by this will be scaled down. New ReplicationControllers will be
  // created with this selector, with a unique label `pod-template-hash`.
  // If Selector is empty, it is defaulted to the labels present on the Pod template.
  Selector map[string]string

  // Describes the pods that will be created.
  Template *PodTemplateSpec

  // The deployment strategy to use to replace existing pods with new ones.
  Strategy DeploymentStrategy
}

type DeploymentStrategy struct {
  // Type of deployment. Can be "Recreate" or "RollingUpdate".
  Type DeploymentStrategyType

  // TODO: Update this to follow our convention for oneOf, whatever we decide it
  // to be.
  // Rolling update config params. Present only if DeploymentStrategyType =
  // RollingUpdate.
  RollingUpdate *RollingUpdateDeploymentStrategy
}

type DeploymentStrategyType string

const (
  // Kill all existing pods before creating new ones.
  RecreateDeploymentStrategyType DeploymentStrategyType = "Recreate"

  // Replace the old RCs by new one using rolling update i.e gradually scale down the old RCs and scale up the new one.
  RollingUpdateDeploymentStrategyType DeploymentStrategyType = "RollingUpdate"
)

// Spec to control the desired behavior of rolling update.
type RollingUpdateDeploymentStrategy struct {
  // The maximum number of pods that can be unavailable during the update.
  // Value can be an absolute number (ex: 5) or a percentage of total pods at the start of update (ex: 10%).
  // Absolute number is calculated from percentage by rounding up.
  // This can not be 0 if MaxSurge is 0.
  // By default, a fixed value of 1 is used.
  // Example: when this is set to 30%, the old RC can be scaled down by 30%
  // immediately when the rolling update starts. Once new pods are ready, old RC
  // can be scaled down further, followed by scaling up the new RC, ensuring
  // that at least 70% of original number of pods are available at all times
  // during the update.
  MaxUnavailable IntOrString

  // The maximum number of pods that can be scheduled above the original number of
  // pods.
  // Value can be an absolute number (ex: 5) or a percentage of total pods at
  // the start of the update (ex: 10%). This can not be 0 if MaxUnavailable is 0.
  // Absolute number is calculated from percentage by rounding up.
  // By default, a value of 1 is used.
  // Example: when this is set to 30%, the new RC can be scaled up by 30%
  // immediately when the rolling update starts. Once old pods have been killed,
  // new RC can be scaled up further, ensuring that total number of pods running
  // at any time during the update is atmost 130% of original pods.
  MaxSurge IntOrString

  // Minimum number of seconds for which a newly created pod should be ready
  // without any of its container crashing, for it to be considered available.
  // Defaults to 0 (pod will be considered available as soon as it is ready)
  MinReadySeconds int
}

type DeploymentStatus struct {
  // Total number of ready pods targeted by this deployment (this
  // includes both the old and new pods).
  Replicas int

  // Total number of new ready pods with the desired template spec.
  UpdatedReplicas int
}

```

### Controller

#### Deployment Controller

The DeploymentController will make Deployments happen.
It will watch Deployment objects in etcd.
For each pending deployment, it will:

1. Find all RCs whose label selector is a superset of DeploymentSpec.Selector.
   - For now, we will do this in the client - list all RCs and then filter the
     ones we want. Eventually, we want to expose this in the API.
2. The new RC can have the same selector as the old RC and hence we add a unique
   selector to all these RCs (and the corresponding label to their pods) to ensure
   that they do not select the newly created pods (or old pods get selected by
   new RC).
   - The label key will be "pod-template-hash".
   - The label value will be hash of the podTemplateSpec for that RC without
     this label. This value will be unique for all RCs, since PodTemplateSpec should be unique.
   - If the RCs and pods dont already have this label and selector:
     - We will first add this to RC.PodTemplateSpec.Metadata.Labels for all RCs to
       ensure that all new pods that they create will have this label.
     - Then we will add this label to their existing pods and then add this as a selector
       to that RC.
3. Find if there exists an RC for which value of "pod-template-hash" label
   is same as hash of DeploymentSpec.PodTemplateSpec. If it exists already, then
   this is the RC that will be ramped up. If there is no such RC, then we create
   a new one using DeploymentSpec and then add a "pod-template-hash" label
   to it. RCSpec.replicas = 0 for a newly created RC.
4. Scale up the new RC and scale down the olds ones as per the DeploymentStrategy.
   - Raise an event if we detect an error, like new pods failing to come up.
5. Go back to step 1 unless the new RC has been ramped up to desired replicas
   and the old RCs have been ramped down to 0.
6. Cleanup.

DeploymentController is stateless so that it can recover in case it crashes during a deployment.

### MinReadySeconds

We will implement MinReadySeconds using the Ready condition in Pod. We will add
a LastTransitionTime to PodCondition and update kubelet to set Ready to false,
each time any container crashes. Kubelet will set Ready condition back to true once
all containers are ready. For containers without a readiness probe, we will
assume that they are ready as soon as they are up.
https://github.com/kubernetes/kubernetes/issues/11234 tracks updating kubelet
and https://github.com/kubernetes/kubernetes/issues/12615 tracks adding
LastTransitionTime to PodCondition.

## Changing Deployment mid-way

### Updating

Users can update an ongoing deployment before it is completed.
In this case, the existing deployment will be stalled and the new one will
begin.
For ex: consider the following case:
- User creates a deployment to rolling-update 10 pods with image:v1 to
  pods with image:v2.
- User then updates this deployment to create pods with image:v3,
  when the image:v2 RC had been ramped up to 5 pods and the image:v1 RC
  had been ramped down to 5 pods.
- When Deployment Controller observes the new deployment, it will create
  a new RC for creating pods with image:v3. It will then start ramping up this
  new RC to 10 pods and will ramp down both the existing RCs to 0.

### Deleting

Users can pause/cancel a deployment by deleting it before it is completed.
Recreating the same deployment will resume it.
For ex: consider the following case:
- User creates a deployment to rolling-update 10 pods with image:v1 to
  pods with image:v2.
- User then deletes this deployment while the old and new RCs are at 5 replicas each.
  User will end up with 2 RCs with 5 replicas each.
User can then create the same deployment again in which case, DeploymentController will
notice that the second RC exists already which it can ramp up while ramping down
the first one.

### Rollback

We want to allow the user to rollback a deployment. To rollback a
completed (or ongoing) deployment, user can create (or update) a deployment with
DeploymentSpec.PodTemplateSpec = oldRC.PodTemplateSpec.

## Deployment Strategies

DeploymentStrategy specifies how the new RC should replace existing RCs.
To begin with, we will support 2 types of deployment:
* Recreate: We kill all existing RCs and then bring up the new one. This results
  in quick deployment but there is a downtime when old pods are down but
  the new ones have not come up yet.
* Rolling update: We gradually scale down old RCs while scaling up the new one.
  This results in a slower deployment, but there is no downtime. At all times
  during the deployment, there are a few pods available (old or new). The number
  of available pods and when is a pod considered "available" can be configured
  using RollingUpdateDeploymentStrategy.

In future, we want to support more deployment types.

## Future

Apart from the above, we want to add support for the following:
* Running the deployment process in a pod: In future, we can run the deployment process in a pod. Then users can define their own custom deployments and we can run it using the image name.
* More DeploymentStrategyTypes: https://github.com/openshift/origin/blob/master/examples/deployment/README.md#deployment-types lists most commonly used ones.
* Triggers: Deployment will have a trigger field to identify what triggered the deployment. Options are: Manual/UserTriggered, Autoscaler, NewImage.
* Automatic rollback on error: We want to support automatic rollback on error or timeout.

## References

- https://github.com/kubernetes/kubernetes/issues/1743 has most of the
  discussion that resulted in this proposal.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/deployment.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
