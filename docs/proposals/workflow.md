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
[here](http://releases.k8s.io/release-1.1/docs/proposals/workflow.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal to modify [`Job` resource](../../docs/user-guide/jobs.md) to implement a minimal [Workflow managment system](https://en.wikipedia.org/wiki/Workflow_management_system) in kubernetes.
Workflows (aka DAG workflows since jobs are organized in a Direct Acyclic Graph) are ubiquitous in modern [job schedulers](https://en.wikipedia.org/wiki/Job_scheduler), see for example:
    - [luigi](https://github.com/spotify/luigi)
    - [ozie](http://oozie.apache.org/)
    - [azkaban](https://azkaban.github.io/)

Most of the [job schedulers](https://en.wikipedia.org/wiki/List_of_job_scheduler_software) offer workflow functionality to some extent.

## Use Cases

    * As a user I want to be able to define job chains such that the completion of a given job will trigger other jobs.

## Implementation

The basic idea consists in adding a label selector to the current Job API object. The new selector will determine the parent jobs.  The parent jobs are the jobs current job will depend on. The current job will be scheduled once all the parent jobs will run to completion.
The strongest point in favor of this approach is the ability to re-use the current `job` implementation. Using a label selector the current `job` will simply become a `graph` of jobs: all the features/controllers that will be built on top of `job` like for example [#11980](https://github.com/kubernetes/kubernetes/issues/11980) and [#17242](https://github.com/kubernetes/kubernetes/issues/17242) will be automatically extended to support workflows. Not sure about https://github.com/kubernetes/kubernetes/issues/16845 and https://github.com/kubernetes/kubernetes/issues/14188.
A similar approach is implemented in Chronos for [dependent jobs](https://mesos.github.io/chronos/docs/api.html#adding-a-dependent-job).

### API Object

To implement _workflows_ the `Job` API should be modified:

```go
// JobsSelector it's just an alias for PodSelector
type JobSelector PodSelector

// Job represents the configuration of a single job.
type Job struct {
    unversioned.TypeMeta `json:",inline"`
    // Standard object's metadata.

    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
    api.ObjectMeta `json:"metadata,omitempty"`

    // Spec is a structure defining the expected behavior of a job.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
    Spec JobSpec `json:"spec,omitempty"`

    // Status is a structure describing current status of a job.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
    Status JobStatus `json:"status,omitempty"`

    // JobSelector to detect parents jobs
    ParentSelector *JobSelector `json:"parentSelector, omitempty"`
}
```

#### JobSpec and JobStatus

No modifications

#### Parent JobSelector

The `.job.parentselector` is a label query over a set of jobs.
If all selected jobs are completed the `job` will be started immediately.
If `.job.parentselector` is absent the `job` will be started immediately.
Since labels are forwarded directly from `job` to pods, users should not create pods whose labels match this selector, either directly,
via another Job, or via another controller (for example Replication controller), [see](https://github.com/kubernetes/kubernetes/issues/14961).

#### Some special cases

Jobs terminated via [#17244](https://github.com/kubernetes/kubernetes/issues/17244)  (if implemented) will be considered _completed_ so will trigger the _child_ jobs. Since it's very unlikely that the user would keep the _child_ jobs alive we may add a field to cascade termination to _child_ jobs if any to the current proposals made by @pmorie and @bgrant0607.

For example

```json
apiVersion: v1alpha1
kind: UpwardAPIRequest
spec:
    terminateExistingJob: true
    reason: "TimeOut"
    message: "Pods ran out of time"
    cascade: true
```

### CRUD

    - Creating a job without an empty `jobs.parentselector` will simply follow the usual `job` life cycle.  With a non non nil `jobs.parentselector` the logic already described will be considered.
    - Users may read the `job` using the usual `get`, `describe` commands already implemented for `job`. Obviously the `describe` command should display the non nil `jobs.parentsselector` (if any).
    - A `job` cannot be updated. To update a `job`, user should delete and re-create it. The only exception is `job` _scaling_. A `workflow job` (i.e. with a non nil selector) can be scaled in the common way. No matter if pods are currently running or not.
    - A `job` can be deleted in the usual way `job`. A cascade mechanism may be implemented in a later phase.

## Events

The usual Job controller events will be emitted.
    * JobStart
    * JobFinished
Since a Job with a non nil job selector can be created and may never start we propose to add the event:
    * JobCreated

## Known drawbacks

    * Using only a label selector won't permit to implement backtracking for failures, a common functionality for a DAG worflow system. In this cases [controllerRef](https://github.com/kubernetes/kubernetes/issues/2210#issuecomment-134878215) could help.
    * There's no guarantee to DAG rules aren't violated. For example we cannot prevent cycles between one or more jobs. The only check one may perform is to ensure there is no self-loop on each Job (basically preventing a Job from being its own parent).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/workflow.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
