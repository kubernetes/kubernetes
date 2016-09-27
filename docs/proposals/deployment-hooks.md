
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
[here](http://releases.k8s.io/release-1.4/docs/proposals/deployment-hooks.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Deployment Lifecycle Hooks

## Abstract

A proposal for implementing a new feature for the - Deployment - resource which will
enable users to execute arbitrary commands necessary to complete a deployment.

The deployment lifecycle hook is a materialization of the "process" a deployment requires.
A hook is a way of reducing the cost of implementing a full custom deployment process -
instead, at logical points in the flow control of the process is handed off to the control
of user code. Practically, hooks often involve one way transitions in state, such as a
forward database migration, removal of old values from a persistent store, or the clearing
of a state cache. A hook is effectively a synchronous event listener with veto power - it
may return success or failure, or in some cases need reexectution.

Because deployment processes typically involve coupling between state (assumed elsewhere)
and code (frequently the target of the deployment), it should be easy for a hook to be
coupled to a particular version of code, and easy for a deployer to use the code under
deployment from the hook.

## Use Cases

* As a Rails application developer, I want to perform a Rails database migration during
  the application deployment.
* As an application developer, I want to invoke my test suite before the deployment is
  rollout.
* As an application developer, I want to invoke a cloud API endpoint to notify it of the
  presence of new code.

### Use Case: Invoke a Cloud API Endpoint

Consider an application whose deployment should result in a cloud API call being invoked
to notify it of the newly deployed code.

Kubernetes provides container lifecycle hooks for containers within a Pod. Currently,
post-start and pre-stop hooks are supported. For deployments, post-start is the most
relevant. Because these hooks are implemented by the Kubelet, the post-start hook provides
some unique guarantees:

* The hook is executed synchronously during the pod lifecycle.
* The status of the pod is linked to the status and outcome of the hook execution.
* The pod will not enter a ready status until the hook has completed successfully.
* Service endpoints will not be created for the pod until the pod has a ready status.
* If the hook fails, the pod's creation is considered a failure, and the retry behavior is
  restart-policy driven in the usual way.
* Because deployments are represented as replication controllers, lifecycle hooks defined
  for containers are executed for every container in the replica set for the deployment.
  This behavior has complexity implications when applied to deployment use cases:

The hooks for all pods in the deployment will race, placing a burden on hook authors
(e.g., the hooks would generally need to be tolerant of concurrent hook execution and
implement manual coordination.)

#### Container lifecycle hooks

Container lifecycle hooks aren't ideal for this use case because they will be fired once
per pod in the deployment during scale-up rather than following the logical deployment as
a whole. Consider an example deployment flow using container lifecycle hooks:

1. Deployment is created.
2. Deployment is scaled up to 10 by the deployment strategy.
3. The cloud API is invoked 10 times.
4. Deployment is considered complete concurrently with the cloud API calls.
5. Deployment is scaled up to 15 to handle increased application traffic.

The cloud API is invoked 5 times, outside the deployment workflow.

#### Deployment hooks

A post-deployment hook would satisfy the use case by ensuring the API call is invoked
after the deployment has been rolled out. For example, the flow of this deployment would
be:

1. Deployment is created.
2. Deployment is scaled up to 10 by the deployment strategy.
3. Deployment hook fires, invoking the cloud API.
4. Deployment is considered complete.
5. Deployment is scaled up to 15 to handle increased application traffic.
6. No further calls to cloud API are made until next deployment.

## Proposed Design

### Lifecycle Pods

The definition of a lifecycle hooks is similar to a Pod but instead of specifying the
image, you  you reference a container name from the DeploymentSpec template whose image
will be used to create a container where the command defined in lifecycle hook will run.

Users might also optionally define additional environment variables to adjust the
environment for the hook command (eg. test variables, database schema version, etc..)

Users might also optionally specify a list of volume names that reference the volumes
defined in the DeploymentSpec. These volumes will be then bound to a lifecycle Pod and
will be accessible during the hook execution time. 

### Stages

We propose adding three stages to reflect the deployment lifecycle:

* **pre** - a hook that is executed _before_ the application is rollout, iow. before the
  strategy manipulates the deployment. This hooks can be used to migrate the existing
  application data or perform database backups. It also can be used to signal other
  applications or services that this application is going to be updated.

* **mid** - a hook that is executed in the _middle_ of the the application rollout, iow.
  while the deployment is scaled down to zero before the first new pod is created. This
  only works for the Recreate strategy. This hook can be used to migrate the application
  database (Rails) as the old version of the application is inactive (scaled to 0) and the
  new version is about to be deployed.

* **post** - a hook that is executed _after_ the application rollout, iow. after the
  strategy finishes all the deployment logic. This hook can be used to verify the end
  state of the deployment or signal the other services that the update of the application
  is complete.

### Handling Failures

Each lifecycle hook should have configurable behavior in case of a failure occurs. The
failure is determined by the state of the Pod where the lifecycle command is executed.
Definition of the lifecycle hook can set one of these `FailurePolicy`:

* **Abort** - using this policy will cause the deployment process to abort and set
  the condition to `Failure` in case the lifecycle Pod exited with non-zero code. In the
  context of deployment, these hooks are considered as "mandatory" and they failure have
  impact on the deployment progress.

* **Ignore** - using this policy will cause the deployment process to continue
  regardless of the exit status of a lifecycle Pod.

* **Retry** - using this policy will cause that in case of a deployment lifecycle
  hook failure, the lifecycle Pod will be restarted until the deployment reached the
  `ProgressDeadlineSeconds` or the hook succeeds.

If a new deployment is started in middle of the rollout or while the deployment lifecycle
hook is being executed the lifecycle Pod will be aborted and the new rollout will be
triggered. Note that this might lead in data corruption if the hooks are used to migrate
application data to database and users are advised not mutate the deployment spec while
another deployment is making progress.

Every failure of a lifecycle hook should be logged as an Event regardless of the failure
policy.

### Default values

Initially the default `FailurePolicy` should be set to `Retry` until the deployments can
safely be [rolled back automatically](https://github.com/kubernetes/kubernetes/issues/23211).

### Associating lifecycle Pods to Deployment

The `OwnerReference` should be considered to be set on lifecycle Pods to a ReplicaSet that
represents the deployment they were created for. This allows to garbage collect when the
ReplicaSet is deleted.

### Garbage Collection

After a successfull deployment the lifecycle hook pods should be considered for garbage
collection. It it questionable if the failed lifecycle hooks should be opted-out from
garbage collection to allow future inspection.

## API

The following structures will be added:

```go
type LifecycleHook struct {
  // FailurePolicy specifies what action to take if the hook fails.
  FailurePolicy LifecycleHookFailurePolicy

  // ExecNewPod specifies the options for a lifecycle hook backed by a pod.
  ExecNewPod *ExecNewPodHook
}
```

```go
type LifecycleHookFailurePolicy string

const (
  // LifecycleHookFailurePolicyRetry means retry the hook until it succeeds.
  LifecycleHookFailurePolicyRetry LifecycleHookFailurePolicy = "Retry"

  // LifecycleHookFailurePolicyAbort means abort the deployment (if possible).
  LifecycleHookFailurePolicyAbort LifecycleHookFailurePolicy = "Abort"

  // LifecycleHookFailurePolicyIgnore means ignore failure and continue the deployment.
  LifecycleHookFailurePolicyIgnore LifecycleHookFailurePolicy = "Ignore"
)
```

```go
type ExecNewPodHook struct {
  // Command specifies the ENTRYPOINT of the container image
  Command []string

  // Args specifies the CMD of the container image
  Args []string

  // Env is a set of environment variables to supply to the hook pod's container.
  Env []api.EnvVar

  // ContainerName is the name of a container in the deployment pod template
  // whose Docker image will be used for the hook pod's container.
  ContainerName string

  // Volumes is a list of named volumes from the pod template which should be
  // copied to the hook pod. Volumes names not found in pod spec are ignored.
  // An empty list means no volumes will be copied.
  Volumes []string
}
```

The `RollingUpdateDeployment` API object will add these fields:

```go
type RollingUpdateDeployment struct {
  // Pre is a lifecycle hook which is executed before the deployment process
  // begins. All LifecycleHookFailurePolicy values are supported.
  Pre *LifecycleHook

  // Post is a lifecycle hook which is executed after the strategy has
  // finished all deployment logic. The LifecycleHookFailurePolicyAbort policy
  // is NOT supported.
  Post *LifecycleHook
}
```

The `RecreateUpdateDeployment` API object will be created:

```go
type RecreateUpdateDeployment struct {
  // Pre is a lifecycle hook which is executed before the strategy manipulates
  // the deployment. All LifecycleHookFailurePolicy values are supported.
  Pre *LifecycleHook

  // Mid is a lifecycle hook which is executed while the deployment is scaled down to zero before the first new
  // pod is created. All LifecycleHookFailurePolicy values are supported.
  Mid *LifecycleHook

  // Post is a lifecycle hook which is executed after the strategy has
  // finished all deployment logic. All LifecycleHookFailurePolicy values are supported.
  Post *LifecycleHook
}
```

## Controller Changes

The DeploymentController will manage the deployment lifecycle hook spawning.

For each pending deployment, it will:

1. Find all defined lifecycle hooks, based on the deployment strategy used.
2. If a `pre` hook is defined, it creates a new ReplicaSet (`rs1`) with zero replicas.
   Before the old ReplicasSet (`rs0`) is scaled to zero, it executes the `pre`
   lifecycle hook. If the failure policy is set to `Abort` and the lifecycle exits with
   non-zero status, the deployment condition is set to `Failed` and the rollout is
   aborted.
3. If a `mid` hook is defined and the strategy is `Recreate`, it scales the `rs0` to zero
   and then execute the the `mid` lifecycle hook. Same rules about the failure policy
   applies.
4. If a `post` hook is defined, the `rs1` is scaled to its targed replica count and
   the `post` lifecycle hook is executed. Same rules about the failure policy applies.

## References

- https://github.com/kubernetes/kubernetes/issues/14512 contains the original RFE
- https://github.com/openshift/origin/blob/master/docs/proposals/post-deployment-hooks.md contains the OpenShift hooks proposal
