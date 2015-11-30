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

* [luigi](https://github.com/spotify/luigi)
* [ozie](http://oozie.apache.org/)
* [azkaban](https://azkaban.github.io/)

Most of the [job schedulers](https://en.wikipedia.org/wiki/List_of_job_scheduler_software) offer workflow functionality to some extent.

## Use Cases

* As a user I want to be able to define job chains such that the completion of a given job will trigger other jobs.

## Implementation

The basic idea is to add a label selector to the current Job API object. The new selector will determine the parent jobs.  The parent jobs are the jobs current job will depend on. The current job will be scheduled once all the parent jobs will run to completion.
The strongest point in favor of this approach is the ability to re-use the current `job` implementation. Using a label selector the current `job` will simply become a `graph` of jobs: all the features/controllers that will be built on top of `job` like for example [#11980](https://github.com/kubernetes/kubernetes/issues/11980) and [#17242](https://github.com/kubernetes/kubernetes/issues/17242) will be automatically extended to support workflows. Not sure about [#16845](https://github.com/kubernetes/kubernetes/issues/16845) and [#14188](https://github.com/kubernetes/kubernetes/issues/14188).
A similar approach is implemented in Chronos for [dependent jobs](https://mesos.github.io/chronos/docs/api.html#adding-a-dependent-job).

### API Object

To implement _workflows_ the `Job` API should be modified:

```go
// JobSpec describes how the job execution will look like.

// These are valid conditions of a job.
const (
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
	// JobWaiting means the job is waiting for its parents to finish their tasks.
	JobWaiting  JobConditionType = "Waiting"
)

type JobSpec struct {
	...
	// Job labels selector to detect parent jobs.
	ParentSelector map[string]string `json:"parentSelector"`
}

```

#### JobSpec

A new labels selector will be added to `job.spec`. The `job.Spec.ParentSelector` is a label query over a set of jobs.
If all selected jobs are completed the `job` will be started immediately.
If `job.Spec.ParentSelector` is absent the `job` will be started immediately.
If `job.Spec.ParentSelector` is pointing to a non-existing job, the job will wait indefinitely.
Since labels are forwarded directly from `job` to pods, users should not create pods whose labels match this selector, either directly,
via another Job, or via another controller (for example Replication controller), [see](https://github.com/kubernetes/kubernetes/issues/14961).

#### JobStatus

`job.Status` won't be modified, but a new job condition will be added.

#### JobCondition

A new constant value will be added to `JobConditionType` - `Waiting`. This will inform the job is waiting for its parents to finish execution. `Waiting` condition will be valid only for `workflow job`.

### CRUD

* Creating a job without a `job.Spec.Parentselector` (nil `job.Spec.Parentselector`) will simply follow the usual `job` life cycle.
* Without a `job.Spec.ParentSelector` (i.e. nil) the job will be started immediately.
* With a non nil `job.Spec.ParentSelector` the logic already described will be considered.
* The validation will prevent the user to set an _empty_ `job.Spec.ParentSelector`.
* Users may read the `job` using the usual `get`, `describe` commands already implemented for `job`. The `describe` command should display the non nil `job.Spec.ParentSelector` (if any).
* A `job` cannot be updated. To update a `job`, user should delete and re-create it. The only exception is `job` _scaling_. A `workflow job` (i.e. with a non nil selector) can be scaled in the common way. No matter if pods are currently running or not.
* A `job` can be deleted in the usual way.

## Events

The usual Job controller events will be emitted.

* JobStart
* JobFinished

Since a Job with a non nil job selector can be created and may never start we propose to add the events:

* JobWaiting

## Known drawbacks

* Using only a label selector and a boolean field will produce only a very limited ability to troubleshoot failures (for example backtracking chain of jobs in case of failures). In this cases [controllerRef](https://github.com/kubernetes/kubernetes/issues/2210#issuecomment-134878215) could help.
* No guarantee DAG rules


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/workflow.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
