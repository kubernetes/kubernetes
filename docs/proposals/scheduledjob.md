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
1. Be able to create a periodic job, e.g. database backup, sending emails.


## Motivation

ScheduledJobs are needed for performing all time-related actions, namely backups,
report generation and the like.  Each of these tasks should be allowed to run
repeatedly (once a day/month, etc.) or once at a given point in time.


## Design Overview

Users create a ScheduledJob object.  One ScheduledJob object
is like one line of a crontab file.  It has a schedule of when to run,
in [Cron](https://en.wikipedia.org/wiki/Cron) format.


The ScheduledJob controller creates a Job object [Job](job.md)
about once per execution time of the scheduled (e.g. once per
day for a daily schedule.)  We say "about" because there are certain
circumstances where two jobs might be created, or no job might be
created.  We attempt to make these rare, but do not completely prevent
them.  Therefore, Jobs should be idempotent.

The Job object is responsible for any retrying of Pods, and any parallelism
among pods it creates, and determining the success or failure of the set of
pods.  The ScheduledJob does not examine pods at all.


### ScheduledJob resource

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

Users must use a generated selector for the job.

## Modifications to Job resource

TODO for beta: forbid manual selector since that could cause confusing between
subsequent jobs.

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

## Fields Added to Job Template

When the controller creates a Job from the JobTemplateSpec in the ScheduledJob, it
adds the following fields to the Job:

- a name, based on the ScheduledJob's name, but with a suffix to distinguish
  multiple executions, which may overlap.
- the standard created-by annotation on the Job, pointing to the SJ that created it
  The standard key is `kubernetes.io/created-by`.  The value is a serialized JSON object, like
  `{ "kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"ScheduledJob","namespace":"default",`
  `"name":"nightly-earnings-report","uid":"5ef034e0-1890-11e6-8935-42010af0003e","apiVersion":...`
  This serialization contains the UID of the parent.  This is used to match the Job to the SJ that created
  it.

## Updates to ScheduledJobs

If the schedule is updated on a ScheduledJob, it will:
- continue to use the Status.Active list of jobs to detect conflicts.
- try to fulfill all recently-passed times for the new schedule, by starting
  new jobs.  But it will not try to fulfill times prior to the
  Status.LastScheduledTime.
  - Example:   If you have a schedule to run every 30 minutes, and change that to hourly, then the previously started
    top-of-the-hour run, in Status.Active, will be seen and no new job started.
  - Example:   If you have a schedule to run every hour, change that to 30-minutely, at 31 minutes past the hour,
    one run will be started immediately for the starting time that has just passed.

If the job template of a ScheduledJob is updated, then future executions use the new template
but old ones still satisfy the schedule and are not re-run just because the template changed.

If you delete and replace a ScheduledJob with one of the same name, it will:
- not use any old Status.Active, and not consider any existing running or terminated jobs from the previous
  ScheduledJob (with a different UID) at all when determining coflicts, what needs to be started, etc.
- If there is an existing Job with the same time-based hash in its name (see below), then
  new instances of that job will not be able to be created.  So, delete it if you want to re-run.
with the same name as conflicts.
- not "re-run" jobs for "start times" before the creation time of the new ScheduledJobJob object.
- not consider executions from the previous UID when making decisions about what executions to
 start, or status, etc.
- lose the history of the old SJ.

To preserve status, you can suspend the old one, and make one with a new name, or make a note of the old status.


## Fault-Tolerance

### Starting Jobs in the face of controller failures

If the process with the scheduledJob controller in it fails,
and takes a while to restart, the scheduledJob controller
may miss the time window and it is too late to start a job.

With a single scheduledJob controller process, we cannot give
very strong assurances about not missing starting jobs.

With a suggested HA configuration, there are multiple controller
processes, and they use master election to determine which one
is active at any time.

If the Job's StartingDeadlineSeconds is long enough, and the
lease for the master lock is short enough, and other controller
processes are running, then a Job will be started.

TODO: consider hard-coding the minimum StartingDeadlineSeconds
at say 1 minute.  Then we can offer a clearer guarantee,
assuming we know what the setting of the lock lease duration is.

### Ensuring jobs are run at most once

There are three problems here:

- ensure at most one Job created per "start time" of a schedule.
- ensure that at most one Pod is created per Job
- ensure at most one container start occurs per Pod

#### Ensuring one Job

Multiple jobs might be created in the following sequence:

1. scheduled job controller sends request to start Job J1 to fulfill start time T.
1. the create request is accepted by the apiserver and enqueued but not yet written to etcd.
1. scheduled job controller crashes
1. new scheduled job controller starts, and lists the existing jobs, and does not see one created.
1. it creates a new one.
1. the first one eventually gets written to etcd.
1. there are now two jobs for the same start time.

We can solve this in several ways:

1. with three-phase protocol, e.g.:
  1. controller creates a "suspended" job.
  1. controller writes writes an annotation in the SJ saying that it created a job for this time.
  1. controller unsuspends that job.
1. by picking a deterministic name, so that at most one object create can succeed.

#### Ensuring one Pod

Job object does not currently have a way to ask for this.
Even if it did, controller is not written to support it.
Same problem as above.

#### Ensuring one container invocation per Pod

Kubelet is not written to ensure at-most-one-container-start per pod.

#### Decision

This is too hard to do for the alpha version.  We will await user
feedback to see if the "at most once" property is needed in the beta version.

This is awkward but possible for a containerized application ensure on it own, as it needs
to know what ScheduledJob name and Start Time it is from, and then record the attempt
in a shared storage system.   We should ensure it could extract this data from its annotations
using the downward API.

## Name of Jobs

A ScheduledJob creates one Job at each time when a Job should run.
Since there may be concurrent jobs, and since we might want to keep failed
non-overlapping Jobs around as a debugging record, each Job created by the same ScheduledJob
needs a distinct name.

To make the Jobs from the same ScheduledJob distinct, we could use a random string,
in the way that pods have a `generateName`.  For example, a scheduledJob named `nightly-earnings-report`
in namespace `ns1` might create a job `nightly-earnings-report-3m4d3`, and later create
a job called `nightly-earnings-report-6k7ts`.  This is consistent with pods, but
does not give the user much information.

Alternatively, we can use time as a uniqifier.  For example, the same scheduledJob could
create a job called `nightly-earnings-report-2016-May-19`.
However, for Jobs that run more than once per day, we would need to represent
time as well as date.  Standard date formats (e.g. RFC 3339) use colons for time.
Kubernetes names cannot include time.  Using a non-standard date format without colons
will annoy some users.

Also, date strings are much longer than random suffixes, which means that
the pods will also have long names, and that we are more likely to exceed the
253 character name limit when combining the scheduled-job name,
the time suffix, and pod random suffix.

One option would be to compute a hash of the nominal start time of the job,
and use that as a suffix.  This would not provide the user with an indication
of the start time, but it would prevent creation of the same execution
by two instances (replicated or restarting) of the controller process.

We chose to use the hashed-date suffix approach.

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
