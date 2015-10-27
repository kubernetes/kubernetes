<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Job Controller

## Abstract

A proposal for implementing a new controller - Job controller - which will be responsible
for managing pod(s) that require running once to completion even if the machine
the pod is running on fails, in contrast to what ReplicationController currently offers.

Several existing issues and PRs were already created regarding that particular subject:
* Job Controller [#1624](https://github.com/kubernetes/kubernetes/issues/1624)
* New Job resource [#7380](https://github.com/kubernetes/kubernetes/pull/7380)


## Use Cases

1. Be able to start one or several pods tracked as a single entity.
1. Be able to run batch-oriented workloads on Kubernetes.
1. Be able to get the job status.
1. Be able to specify the number of instances performing a job at any one time.
1. Be able to specify the number of successfully finished instances required to finish a job.


## Motivation

Jobs are needed for executing multi-pod computation to completion; a good example
here would be the ability to implement any type of batch oriented tasks.


## Implementation

Job controller is similar to replication controller in that they manage pods.
This implies they will follow the same controller framework that replication
controllers already defined.  The biggest difference between a `Job` and a
`ReplicationController` object is the purpose; `ReplicationController`
ensures that a specified number of Pods are running at any one time, whereas
`Job` is responsible for keeping the desired number of Pods to a completion of
a task.  This difference will be represented by the `RestartPolicy` which is
required to always take value of `RestartPolicyNever` or `RestartOnFailure`.


The new `Job` object will have the following content:

```go
// Job represents the configuration of a single job.
type Job struct {
    TypeMeta
    ObjectMeta

    // Spec is a structure defining the expected behavior of a job.
    Spec JobSpec

    // Status is a structure describing current status of a job.
    Status JobStatus
}

// JobList is a collection of jobs.
type JobList struct {
    TypeMeta
    ListMeta

    Items []Job
}
```

`JobSpec` structure is defined to contain all the information how the actual job execution
will look like.

```go
// JobSpec describes how the job execution will look like.
type JobSpec struct {

    // Parallelism specifies the maximum desired number of pods the job should
    // run at any given time. The actual number of pods running in steady state will
    // be less than this number when ((.spec.completions - .status.successful) < .spec.parallelism),
    // i.e. when the work left to do is less than max parallelism.
    Parallelism *int

    // Completions specifies the desired number of successfully finished pods the
    // job should be run with. Defaults to 1.
    Completions *int

    // Selector is a label query over pods running a job.
    Selector map[string]string

    // Template is the object that describes the pod that will be created when
    // executing a job.
    Template *PodTemplateSpec
}
```

`JobStatus` structure is defined to contain informations about pods executing
specified job.  The structure holds information about pods currently executing
the job.

```go
// JobStatus represents the current state of a Job.
type JobStatus struct {
    Conditions []JobCondition

    // CreationTime represents time when the job was created
    CreationTime unversioned.Time

    // StartTime represents time when the job was started
    StartTime unversioned.Time

    // CompletionTime represents time when the job was completed
    CompletionTime unversioned.Time

    // Active is the number of actively running pods.
    Active int

    // Successful is the number of pods successfully completed their job.
    Successful int

    // Unsuccessful is the number of pods failures, this applies only to jobs
    // created with RestartPolicyNever, otherwise this value will always be 0.
    Unsuccessful int
}

type JobConditionType string

// These are valid conditions of a job.
const (
    // JobComplete means the job has completed its execution.
    JobComplete JobConditionType = "Complete"
)

// JobCondition describes current state of a job.
type JobCondition struct {
    Type               JobConditionType
    Status             ConditionStatus
    LastHeartbeatTime  unversioned.Time
    LastTransitionTime unversioned.Time
    Reason             string
    Message            string
}
```

## Events

Job controller will be emitting the following events:
* JobStart
* JobFinish

## Future evolution

Below are the possible future extensions to the Job controller:
* Be able to limit the execution time for a job, similarly to ActiveDeadlineSeconds for Pods.
* Be able to create a chain of jobs dependent one on another.
* Be able to specify the work each of the workers should execute (see type 1 from
  [this comment](https://github.com/kubernetes/kubernetes/issues/1624#issuecomment-97622142))
* Be able to inspect Pods running a Job, especially after a Job has finished, e.g.
  by providing pointers to Pods in the JobStatus ([see comment](https://github.com/kubernetes/kubernetes/pull/11746/files#r37142628)).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/job.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
