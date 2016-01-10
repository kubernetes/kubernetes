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

A proposal to introduce [workflow](https://en.wikipedia.org/wiki/Workflow_management_system)
functionality in kubernetes.
Workflows (aka [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) workflows
since _tasks_ are organized in a Directed Acyclic Graph) are ubiquitous
in modern [job schedulers](https://en.wikipedia.org/wiki/Job_scheduler), see for example:

* [luigi](https://github.com/spotify/luigi)
* [ozie](http://oozie.apache.org/)
* [azkaban](https://azkaban.github.io/)

Most of the [job schedulers](https://en.wikipedia.org/wiki/List_of_job_scheduler_software) offer
workflow functionality to some extent.


## Use Cases

* As a user, I want to create a _JobB_ which depends upon _JobA_ running to completion.
* As a user, I want workflow composability. I want to create a _JobA_ which will be triggered
as soon as an already running workflow runs to completion.
* As a user, I want to delete a workflow (eventually cascading to running _tasks_).
* As a user, I want to debug a workflow (ability to track failure): in case a _task_
didn't run user should have a way to backtrack the reason of the failure, understanding which
dependency has not been satisified.


## Implementation

This proposal introduces a new REST resource `Workflow`. A `Workflow` is represented as a
[graph](https://en.wikipedia.org/wiki/Graph_(mathematics)), more specifically as a DAG.
Vertices of the graph represent steps of the workflow. The workflow steps are represented via a
`WorkflowStep`<sup>1</sup> resource.
The edges of the graph represent _dependecies_. To represent edges there is no explicit resource
- rather they are stored as predecessors in each `WorkflowStep` (i.e. each node).
The basic idea of this proposal consists in creation of each step postponing execution
until all predecessors' steps run to completion.


### Postponing execution

At the time of writing, to defer execution there are two discussions in the community:
[#17305](https://github.com/kubernetes/kubernetes/pull/17305): an
_initializer_ is a dynamically registered object which permits to select a custom controller
to be applied to a resource. The controller verifies the dependencies.
The controller checks are applied before the resource is created (even API validated).
Using a proper controller one may defer creation of the resource until prerequisites
are satisfied. Even if not completed [#17305](https://github.com/kubernetes/kubernetes/pull/17305)
already introduces a _dependency_ concept
([see this comment](https://github.com/kubernetes/kubernetes/pull/17305#discussion_r45007826))
which could be reused to implement `Workflow`. In
[#1899](https://github.com/kubernetes/kubernetes/issues/1899):
some use-cases to wait for specific conditions (`complete`, `ready`) are presented.


### Detecting run to completion

To detect run to completion for the resource inside the graph the resource needs to implement
in `status` the slice of `condition`s. [See](../../docs/devel/api-conventions.md#objects)
and [#7856](https://github.com/kubernetes/kubernetes/issues/7856).

### Workflow

A new resource will be introduced in the API. A `Workflow` is a graph.
In the simplest case it's a graph of `Job`s but it can also
be a graph of other entities (for example cross-cluster objects or other `Workflow`s).


```go
// Workflow is a directed acyclic graph
type Workflow struct {
    unversioned.TypeMeta `json:",inline"`

    // Standard object's metadata.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata.
    api.ObjectMeta `json:"metadata,omitempty"`

    // Spec defines the expected behavior of a Workflow. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
    Spec WorkflowSpec `json:"spec,omitempty"`

    // Status represents the current status of the Workflow. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
    Status WorkflowStatus `json:"status,omitempty"`
}

// WorkflowList implements list of Workflow.
type WorkflowList struct {
    unversioned.TypeMeta `json:",inline"`

    // Standard list metadata
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
    unversioned.ListMeta `json:"metadata,omitempty"`

    // Items is the list of Workflow
    Items []Workflow `json:"items"`
}
```


#### `WorkflowSpec`

```go
// WorkflowSpec contains Workflow specification
type WorkflowSpec struct {
    // Optional duration in seconds relative to the startTime that the job may be active
    // before the system tries to terminate it; value must be positive integer
    ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

    // Steps contains the vertices of the workflow graph. The key of the map is a string
    // to uniquely identify the step. Steps order is defined by their dependencies.
    Steps map[string]WorkflowStep `json:"steps,omitempty"`
}
```

* `spec.steps`: is a map of `WorkflowStep`s. _Key_ of the map is a string which identifies the step.


### `WorkflowStep`

The `WorkflowStep` resource acts as a [union](https://en.wikipedia.org/wiki/Tagged_union) of `JobSpec` and `ObjectReference`.

```go
// WorkflowStep contains necessary information to identifiy the node of the workflow graph
type WorkflowStep struct {
    // JobTemplate contains the job specificaton that should be run in this Workflow.
    // Only one between externalRef and jobTemplate can be set.
    JobTemplate JobSpec `json:"jobTemplate,omitempty"`

    // External contains a reference to another schedulable resource.
    // Only one between ExternalRef and JobTemplate can be set.
    ExternalRef api.ObjectReference `json:"externalRef,omitempty"`

    // Dependecies represent dependecies of the current workflow step
    Dependencies ObjectDependencies `json:"dependencies,omitempty"`
}
```

* `workflowStep.jobSpec` contains the specification of the job to be executed.
* `workflowStep.externalRef` contains a reference to external resources (for example another `Workflow`).
*

```go
type ObjectDependencies struct {
    // DependeciesRef is a slice of unique identifier of the step (key of the spec.steps map)
    DependencyIDs []string `json:"dependencyIDs,omitempty"`
    ControllerRef *ObjectReference `json:"controllerRef,omitempty"`
    //...
}
```

* `dependencies.dependencyIDs`: is a slice with a list of _step_ that must run to completion.
* `dependencies.controllerRef`: will contain the controller for the current `WorkflowStep`. As a first


This approach permits to implement other kinds of controller, for example data availability
or other external event. In a first implementation `dependencies.controllerRef` will implement only
the logic to check all dependencies ran to completion: since at the beginning only `Workflow` and `Job`
can be composed the only thing needed to implement is the ability to check wether a `Job` or
a `Workflow` runs to completion.
Our understanding is that detecting the type of object and an approach similar to what
is implemented in `pkg/client/unversioned/conditions.go` and  `pkg/kubectl/scale.go` for _desiredReplicas_ can
be used to to detect if a _step_ must be started.


### `WorkflowStatus`

```go
// WorkflowStatus contains the current status of the Workflow
type WorkflowStatus struct {
    // Conditions represent the latest available observations of an object's current state.
    Conditions []WorkflowCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

    // Statuses represent status of different steps
    Statuses map[string]WorkflowStepStatus `json:statuses`
}

type WorkflowConditionType string

// These are valid conditions of a workflow.
const (
    // WorkflowComplete means the workflow has completed its execution.
    WorkflowComplete WorkflowConditionType = "Complete"
)

// WorkflowCondition describes current state of a workflow.
type WorkflowCondition struct {
    // Type of workflow condition, currently only Complete.
    Type WorkflowConditionType `json:"type"`
    // Status of the condition, one of True, False, Unknown.
    Status api.ConditionStatus `json:"status"`
    // Last time the condition was checked.
    LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty"`
    // Last time the condition transited from one status to another.
    LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty"`
    // (brief) reason for the condition's last transition.
    Reason string `json:"reason,omitempty"`
    // Human readable message indicating details about last transition.
    Message string `json:"message,omitempty"`
}

// WorkflowStepStatus contains the status of a WorkflowStep
type WorkflowStepStatus struct {
    // ObjectReference contains the reference to the resource
    ObjectReference api.ObjectReference `json:"objectReference,omitempty"`
}
```

* `status.statuses`: is a map of `WorkflowStepStatus`es. _Key_ of the map is a string which identifies the step.
_Keys_ are the same used in `spec.steps`.
* `status.conditions`: is a slice of `WorkflowCondition`s. [see #7856](https://github.com/kubernetes/kubernetes/issues/7856)

## Events

The events associated to `Workflow`s will be:

* WorkflowCreated
* WorkflowStarted
* WorkflowEnded
* WorkflowDeleted


## Future evolution

In the future we may want to extend _Workflow_ with other kinds of resources, modifying `WorkflowStep` to
support a more general template to create other resources.
One of the major functionalities missing here is the ability to set a recurring `Workflow` (cron-like),
similar to the `ScheduledJob` [#11980](https://github.com/kubernetes/kubernetes/pull/11980) for `Job`.
If the scheduled job is able
to support [various resources](https://github.com/kubernetes/kubernetes/pull/11980#discussion_r46729699)
`Workflow` will benefit from the _schedule_ functionality of `Job`.


### Relevant use cases out of scope of this proposal

* As an admin I want to set quota on workflow resources
[#13567](https://github.com/kubernetes/kubernetes/issues/13567).
* As an admin I want to re-assign a workflow resource to another namespace/user<sup>2</sup>.
* As a user I want to set an action when a workflow ends/start
[#3585](https://github.com/kubernetes/kubernetes/issues/3585)


<sup>1</sup>Something about naming: literature is full of different names, a commonly used
name is _task_, but since we plan to compose `Workflow`s (i.e. a task can execute
another whole `Workflow`) we have chosen the more generic word `Step`.


<sup>2</sup>A very common feature in industrial strength workflow tools.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/workflow.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
