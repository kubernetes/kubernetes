<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# ScheduledJob Controller

## Abstract

A proposal for implementing a new controller - ScheduledJob controller - which
will be responsible for managing time based jobs, namely:
* once at a specified point in time,
* repeatedly at a specified point in time.

There is already a discussion regarding this subject:
* Distributed CRON jobs [#2156](https://issues.k8s.io/2156)

There are also similar solutions available, already:
* [Mesos Chronos](https://github.com/mesos/chronos)
* [Quartz](http://quartz-scheduler.org/)


## Use Cases

1. Be able to schedule a job execution at a given point in time.
1. Be able to create a periodic job, eg. database backup, sending emails.


## Motivation

ScheduledJobs are needed for performing all time-related actions, namely backups,
report generation and the like.  Each of these tasks should be allowed to run
repeatedly (once a day/month, etc.) or once at a given point in time.


## Implementation

### ScheduledJob resource

The ScheduledJob controller relies heavily on the [Job API](job.md)
for running actual jobs, on top of which it adds information regarding the date
and time part according to [Cron](https://en.wikipedia.org/wiki/Cron) format.

The new `ScheduledJob` object will have the following contents:

```go
// ScheduledJob represents the configuration of a single scheduled job.
type ScheduledJob struct {
    TypeMeta
    ObjectMeta

    // Spec is a structure defining the expected behavior of a job, including the schedule.
    Spec ScheduledJobSpec

    // Status is a structure describing current status of a job.
    Status ScheduledJobStatus
}

// ScheduledJobList is a collection of scheduled jobs.
type ScheduledJobList struct {
    TypeMeta
    ListMeta

    Items []ScheduledJob
}
```

The `ScheduledJobSpec` structure is defined to contain all the information how the actual
job execution will look like, including the `JobSpec` from [Job API](job.md)
and the schedule in [Cron](https://en.wikipedia.org/wiki/Cron) format.  This implies
that each ScheduledJob execution will be created from the JobSpec actual at a point
in time when the execution will be started.  This also implies that any changes
to ScheduledJobSpec will be applied upon subsequent execution of a job.

```go
// ScheduledJobSpec describes how the job execution will look like and when it will actually run.
type ScheduledJobSpec struct {

    // Schedule contains the schedule in Cron format, see https://en.wikipedia.org/wiki/Cron.
    Schedule string

    // Optional deadline in seconds for starting the job if it misses scheduled
    // time for any reason.  Missed jobs executions will be counted as failed ones.
    StartingDeadlineSeconds *int64

    // ConcurrencyPolicy specifies how to treat concurrent executions of a Job.
    ConcurrencyPolicy ConcurrencyPolicy

    // Suspend flag tells the controller to suspend subsequent executions, it does
    // not apply to already started executions.  Defaults to false.
    Suspend bool

    // JobTemplate is the object that describes the job that will be created when
    // executing a ScheduledJob.
    JobTemplate *JobTemplateSpec
}

// JobTemplateSpec describes of the Job that will be created when executing
// a ScheduledJob, including its standard metadata.
type JobTemplateSpec struct {
    ObjectMeta

    // Specification of the desired behavior of the job.
    Spec JobSpec
}

// ConcurrencyPolicy describes how the job will be handled.
// Only one of the following concurrent policies may be specified.
// If none of the following policies is specified, the default one
// is AllowConcurrent.
type ConcurrencyPolicy string

const (
    // AllowConcurrent allows ScheduledJobs to run concurrently.
    AllowConcurrent ConcurrencyPolicy = "Allow"

    // ForbidConcurrent forbids concurrent runs, skipping next run if previous
    // hasn't finished yet.
    ForbidConcurrent ConcurrencyPolicy = "Forbid"

    // ReplaceConcurrent cancels currently running job and replaces it with a new one.
    ReplaceConcurrent ConcurrencyPolicy = "Replace"
)
```

`ScheduledJobStatus` structure is defined to contain information about scheduled
job executions.  The structure holds a list of currently running job instances
and additional information about overall successful and unsuccessful job executions.

```go
// ScheduledJobStatus represents the current state of a Job.
type ScheduledJobStatus struct {
    // Active holds pointers to currently running jobs.
    Active []ObjectReference

    // Successful tracks the overall amount of successful completions of this job.
    Successful int64

    // Failed tracks the overall amount of failures of this job.
    Failed int64

    // LastScheduleTime keeps information of when was the last time the job was successfully scheduled.
    LastScheduleTime Time
}
```

### Modifications to Job resource

In order to distinguish Job runs, we need to add `UniqueLabelKey` field to `JobSpec`.
This field will be used for creating unique label selectors.

```go
type JobSpec {

    //...

    // Key of the selector that is added to prevent concurrently running Jobs
    // selecting their pods.
    // Users can set this to an empty string to indicate that the system should
    // not add any selector and label. If unspecified, system uses
    // "scheduledjob.kubernetes.io/podTemplateHash".
    // Value of this key is hash of ScheduledJobSpec.PodTemplateSpec.
    // No label is added if this is set to an empty string.
    UniqueLabelKey *string
}
```

Although at Job level empty string is perfectly valid, `ScheduledJob` cannot have
empty selector, it needs to be defined, either by user or generated automatically.
For this to happen, validation will be tightened at ScheduledJob level for this
field to be either nil or non-empty string.

### Running ScheduledJobs using kubectl

A user should be able to easily start a Scheduled Job using `kubectl` (similarly
to running regular jobs). For example to run a job with a specified schedule,
a user should be able to type something simple like:

```
kubectl run pi --image=perl --restart=OnFailure --runAt="0 14 21 7 *" -- perl -Mbignum=bpi -wle 'print bpi(2000)'
```

In the above example:

* `--restart=OnFailure` implies creating a job instead of replicationController.
* `--runAt="0 14 21 7 *"` implies the schedule with which the job should be run, here
  July 7th, 2pm.  This value will be validated according to the same rules which
  apply to `.spec.schedule`.


## Future evolution

Below are the possible future extensions to the Job controller:
* Be able to specify workflow template in `.spec` field. This relates to the work
  happening in [#18827](https://issues.k8s.io/18827).
* Be able to specify more general template in `.spec` field, to create arbitrary
  types of resources. This relates to the work happening in [#18215](https://issues.k8s.io/18215).



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/scheduledjob.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
