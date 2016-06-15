<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->


## Abstract

This proposal introduces [workflow](https://en.wikipedia.org/wiki/Workflow_management_system)
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

In this proposal a new REST resource `Workflow` is introduced. A `Workflow` is represented as a
[graph](https://en.wikipedia.org/wiki/Graph_(mathematics)), more specifically as a DAG.
Vertices of the graph represent steps of the workflow. The workflow steps are represented via a
`WorkflowStep`<sup>1</sup> resource.
The edges of the graph represent _dependecies_. To represent edges there is no explicit resource
- rather they are stored as predecessors in each `WorkflowStep` (i.e. each node).
The basic idea of this proposal consists in creation of each step postponing execution
until all predecessors' steps run to completion.

### Workflow

A new resource will be introduced in the API. A `Workflow` is a graph.
In the simplest case it's a graph of `Job`s but it can also
be a graph of other entities (for example cross-cluster objects or other `Workflow`s).

```go

// Workflow is a DAG workflow
type Workflow struct {
  unversioned.TypeMeta `json:",inline"`

  // Standard object's metadata.
  // More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#metadata.
  api.ObjectMeta `json:"metadata,omitempty"`

  // Spec defines the expected behavior of a Workflow. More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#spec-and-status.
  Spec WorkflowSpec `json:"spec,omitempty"`

  // Status represents the current status of the Workflow. More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#spec-and-status.
  Status WorkflowStatus `json:"status,omitempty"`
}

// WorkflowList implements list of Workflow.
type WorkflowList struct {
  unversioned.TypeMeta `json:",inline"`

  // Standard list metadata
  // More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#metadata
  unversioned.ListMeta `json:"metadata,omitempty"`

  // Items is the list of Workflow
  Items []Workflow `json:"items"`
}

// WorkflowSpec contains Workflow specification
type WorkflowSpec struct {
  // Standard object's metadata.
  // More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#metadata
  api.ObjectMeta `json:"metadata,omitempty"`

  //ActiveDealineSeconds contains
  ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`

  // Steps is a map containing the workflow steps. Key of the
  // map is a string which uniquely identifies the workflow step
  Steps map[string]WorkflowStep `json:"steps,omitempty"`

   // Selector for created jobs (if any)
   Selector *LabelSelector `json:"selector,omitempty"`
}

// WorkflowStep contains necessary information to identifiy the node of the workflow graph
type WorkflowStep struct {
  // JobTemplate contains the job specificaton that should be run in this Workflow.
  // Only one between externalRef and jobTemplate can be set.
  JobTemplate *JobTemplateSpec `json:"jobTemplate,omitempty"`

  // External contains a reference to another schedulable resource.
  // Only one between ExternalRef and JobTemplate can be set.
  ExternalRef *api.ObjectReference `json:"externalRef,omitempty"`

  // Dependecies represent dependecies of the current workflow step.
  // Irrilevant for ExteranlRef step
  Dependencies []string `json:"dependencies,omitempty"`
}

type WorkflowConditionType string

// These are valid conditions of a workflow.
const (
  // WorkflowComplete means the workflow has completed its execution.
  WorkflowComplete WorkflowConditionType = "Complete"
)

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

// WorkflowStatus represents the
type WorkflowStatus struct {
  // Conditions represent the latest available observations of an object's current state.
  Conditions []WorkflowCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

  // StartTime represents time when the job was acknowledged by the Workflow controller
  // It is not guaranteed to be set in happens-before order across separate operations.
  // It is represented in RFC3339 form and is in UTC.
  // StartTime doesn't consider startime of `ExternalReference`
  StartTime *unversioned.Time `json:"startTime,omitempty"`

  // CompletionTime represents time when the workflow was completed. It is not guaranteed to
  // be set in happens-before order across separate operations.
  // It is represented in RFC3339 form and is in UTC.
  CompletionTime *unversioned.Time `json:"completionTime,omitempty"`

  // Statuses represent status of different steps
  Statuses map[string]WorkflowStepStatus `json:statuses`
}

// WorkflowStepStatus contains necessary information for the step status
type WorkflowStepStatus struct {
  //Complete is set to true when resource run to complete
  Complete bool `json:"complete"`

  // Reference of the step resource
  Reference api.ObjectReference `json:"reference"`
}
```

`JobTemplateSpec` is already introduced by
ScheduledJob controller proposal](https://github.com/kubernetes/kubernetes/pull/11980).
Reported for readability:

```go
type JobTemplateSpec struct {
  // Standard object's metadata.
  // More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#metadata
  api.ObjectMeta

  // Spec is a structure defining the expected behavior of a job.
  // More info: http://releases.k8s.io/release-1.3/docs/devel/api-conventions.md#spec-and-status
  Spec JobSpec
}
```

## Controller

Workflow controller will watch `Workflow` objects and any `Job` objects created by the workflow.
the `Job`s objects created in each step.
Each step can contain either another `Workflow` referenced via `workflowStep.ExternalRef`
or a `Job` created via `workflowStep.jobTemplate`.
For each non finished workflow (similarly to Job, Workflow completion is detected iterating
over all the `status.conditions` condition) we check if deadline is not expired.
If deadline expired the workfow is terminated.
If deadline didn't expires the workflow controller iterates over all workflow steps:
   - If step has a status (retrieved via step name (map key) in the `status.statuses`
     map check whether the step is already completed.
   - If step is completed nothing is done.
   - If step is not completed two sub-cases should be analyzed:
     * Step containing workflow: check wether workflow terminated and eventually update
     the `status.statuses[name].complete` entry if applicable
     * Step containing job: check whether job needs to be started or is already started.
       - A step/job is started if it has no dependecies or all its dependencies are already
       terminated. Workflow controller adds some labels to the Job.
       This will permit to obtain the workflow each job belongs to (via `spec.Selector`).
       The step name is equally inserted as a label in each job.
       - If the job is already running, a completion check is performed. If the job terminated
         (checked via conditions `job.status`) the field `status.statusues[name].complete` is updated.
   - When all steps are complete: `complete` condition is added to `status.condition` and the
     `status.completionTime` is updated. At this point, workflow is finished.


## Changing a Workflow

### Updating

User can modify a workflow only if the `step`s under modification are not already running.


### Deleting

Users can cancel a workflow by deleting it before it's completed. All running jobs will be deleted.
Other workflows referenced in steps will not be deleted as they are not owned by the parent workflow.


## Events

The events associated to `Workflow`s will be:

* WorkflowCreated
* WorkflowStarted
* WorkflowEnded
* WorkflowDeleted

## Kubectl

Kubectl will be modified to display workflows. More particularly the `describe` command
will display all the steps with their status. Steps will be topologically sorted and
each dependency will be decorated with its status (wether or not step is waitin for
dependency).

## Future evolution

In the future we may want to extend _Workflow_ with other kinds of resources, modifying `WorkflowStep` to
support a more general template to create other resources.
One of the major functionalities missing here is the ability to set a recurring `Workflow` (cron-like),
similar to the `ScheduledJob` [#11980](https://github.com/kubernetes/kubernetes/pull/11980) for `Job`.
If the scheduled job is able to support
[various resources](https://github.com/kubernetes/kubernetes/pull/11980#discussion_r46729699)
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




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/workflow.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
