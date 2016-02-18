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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/user-guide/jobs.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Jobs

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Jobs](#jobs)
  - [What is a _job_?](#what-is-a-job)
  - [Running an example Job](#running-an-example-job)
  - [Writing a Job Spec](#writing-a-job-spec)
    - [Pod Template](#pod-template)
    - [Pod Selector](#pod-selector)
    - [Parallel Jobs](#parallel-jobs)
      - [Controlling Parallelism](#controlling-parallelism)
  - [Handling Pod and Container Failures](#handling-pod-and-container-failures)
  - [Job Patterns](#job-patterns)
  - [Alternatives](#alternatives)
    - [Bare Pods](#bare-pods)
    - [Replication Controller](#replication-controller)
    - [Single Job starts Controller Pod](#single-job-starts-controller-pod)
  - [Caveats](#caveats)
  - [Future work](#future-work)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _job_?

A _job_ creates one or more pods and ensures that a specified number of them successfully terminate.
As pods successfully complete, the _job_ tracks the successful completions.  When a specified number
of successful completions is reached, the job itself is complete.  Deleting a Job will cleanup the
pods it created.

A simple case is to create 1 Job object in order to reliably run one Pod to completion.
The Job object will start a new Pod if the first pod fails or is deleted (for example
due to a node hardware failure or a node reboot).

A Job can also be used to run multiple pods in parallel.

## Running an example Job

Here is an example Job config.  It computes π to 2000 places and prints it out.
It takes around 10s to complete.
<!-- BEGIN MUNGE: EXAMPLE job.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: Job
metadata:
  name: pi
spec:
  selector:
    matchLabels:
      app: pi
  template:
    metadata:
      name: pi
      labels:
        app: pi
    spec:
      containers:
      - name: pi
        image: perl
        command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
      restartPolicy: Never
```

[Download example](job.yaml?raw=true)
<!-- END MUNGE: EXAMPLE job.yaml -->

Run the example job by downloading the example file and then running this command:

```console
$ kubectl create -f ./job.yaml
job "pi" created
```

Check on the status of the job using this command:

```console
$ kubectl describe jobs/pi
Name:		pi
Namespace:	default
Image(s):	perl
Selector:	app in (pi)
Parallelism:	1
Completions:	1
Start Time:	Mon, 11 Jan 2016 15:35:52 -0800
Labels:		app=pi
Pods Statuses:	0 Running / 1 Succeeded / 0 Failed
No volumes.
Events:
  FirstSeen	LastSeen	Count	From			SubobjectPath	Type		Reason			Message
  ---------	--------	-----	----			-------------	--------	------			-------
  1m		1m		1	{job-controller }			Normal		SuccessfulCreate	Created pod: pi-dtn4q
```

To view completed pods of a job, use `kubectl get pods --show-all`.  The `--show-all` will show completed pods too.

To list all the pods that belong to job in a machine readable form, you can use a command like this:

```console
$ pods=$(kubectl get pods --selector=app=pi --output=jsonpath={.items..metadata.name})
echo $pods
pi-aiw0a
```

Here, the selector is the same as the selector for the job.  The `--output=jsonpath` option specifies an expression
that just gets the name from each pod in the returned list.

View the standard output of one of the pods:

```console
$ kubectl logs $pods
3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989380952572010654858632788659361533818279682303019520353018529689957736225994138912497217752834791315155748572424541506959508295331168617278558890750983817546374649393192550604009277016711390098488240128583616035637076601047101819429555961989467678374494482553797747268471040475346462080466842590694912933136770289891521047521620569660240580381501935112533824300355876402474964732639141992726042699227967823547816360093417216412199245863150302861829745557067498385054945885869269956909272107975093029553211653449872027559602364806654991198818347977535663698074265425278625518184175746728909777727938000816470600161452491921732172147723501414419735685481613611573525521334757418494684385233239073941433345477624168625189835694855620992192221842725502542568876717904946016534668049886272327917860857843838279679766814541009538837863609506800642251252051173929848960841284886269456042419652850222106611863067442786220391949450471237137869609563643719172874677646575739624138908658326459958133904780275901
```

## Writing a Job Spec

As with all other Kubernetes config, a Job needs `apiVersion`, `kind`, and `metadata` fields.  For
general information about working with config files, see [deploying applications](deploying-applications.md),
[configuring containers](configuring-containers.md), and [working with resources](working-with-resources.md) documents.

A Job also needs a [`.spec` section](../devel/api-conventions.md#spec-and-status).

### Pod Template

The `.spec.template` is the only required field of the `.spec`.

The `.spec.template` is a [pod template](replication-controller.md#pod-template).  It has exactly
the same schema as a [pod](pods.md), except it is nested and does not have an `apiVersion` or
`kind`.

In addition to required fields for a Pod, a pod template in a job must specify appropriate
labels (see [pod selector](#pod-selector) and an appropriate restart policy.

Only a [`RestartPolicy`](pod-states.md) equal to `Never` or `OnFailure` are allowed.

### Pod Selector

The `.spec.selector` field is a label query over a set of pods.

The `spec.selector` is an object consisting of two fields:
* `matchLabels` - works the same as the `.spec.selector` of a [ReplicationController](replication-controller.md)
* `matchExpressions` - allows to build more sophisticated selectors by specifying key,
  list of values and an operator that relates the key and values.

When the two are specified the result is ANDed.

If `.spec.selector` is unspecified, `.spec.selector.matchLabels` will be defaulted to
`.spec.template.metadata.labels`.

Also you should not normally create any pods whose labels match this selector, either directly,
via another Job, or via another controller such as ReplicationController.  Otherwise, the Job will
think that those pods were created by it.  Kubernetes will not stop you from doing this.

### Parallel Jobs

There are three main types of jobs:

1. Non-parallel Jobs
  - normally only one pod is started, unless the pod fails.
  - job is complete as soon as Pod terminates successfully.
1. Parallel Jobs with a *fixed completion count*:
  - specify a non-zero positive value for `.spec.completions`
  - the job is complete when there is one successful pod for each value in the range 1 to `.spec.completions`.
  - **not implemented yet:** each pod passed a different index in the range 1 to `.spec.completions`.
1. Parallel Jobs with a *work queue*:
  - do not specify `.spec.completions`
  - the pods must coordinate with themselves or an external service to determine what each should work on
  - each pod is independently capable of determining whether or not all its peers are done, thus the entire Job is done.
  - when _any_ pod terminates with success, no new pods are created.
  - once at least one pod has terminated with success and all pods are terminated, then the job is completed with success.
  - once any pod has exited with success, no other pod should still be doing any work or writing any output.  They should all be
    in the process of exiting.

For a Non-parallel job, you can leave both `.spec.completions` and `.spec.parallelism` unset.  When both are
unset, both are defaulted to 1.

For a Fixed Completion Count job, you should set `.spec.completions` to the number of completions needed.
You can set `.spec.parallelism`, or leave it unset and it will default to 1.

For a Work Queue Job, you must leave `.spec.completions` unset, and set `.spec.parallelism` to
a non-negative integer.

For more information about how to make use of the different types of job, see the [job patterns](#job-patterns) section.


#### Controlling Parallelism

The requested parallelism (`.spec.parallelism`) can be set to any non-negative value.
If it is unspecified, it defaults to 1.
If it is specified as 0, then the Job is effectively paused until it is increased.

A job can be scaled up using the `kubectl scale` command.  For example, the following
command sets `.spec.parallelism` of a job called `myjob` to 10:

```console
$ kubectl scale  --replicas=$N jobs/myjob
job "myjob" scaled
```

You can also use the `scale` subresource of the Job resource.

Actual parallelism (number of pods running at any instant) may be more or less than requested
parallelism, for a variety or reasons:

- For Fixed Completion Count jobs, the actual number of pods running in parallel will not exceed the number of
  remaining completions.   Higher values of `.spec.parallelism` are effectively ignored.
- For work queue jobs, no new pods are started after any pod has succeeded -- remaining pods are allowed to complete, however.
- If the controller has not had time to react.
- If the controller failed to create pods for any reason (lack of ResourceQuota, lack of permission, etc),
  then there may be fewer pods than requested.
- The controller may throttle new pod creation due to excessive previous pod failures in the same Job.
- When a pod is gracefully shutdown, it make take time to stop.


## Handling Pod and Container Failures

A Container in a Pod may fail for a number of reasons, such as because the process in it exited with
a non-zero exit code, or the Container was killed for exceeding a memory limit, etc.  If this
happens, and the `.spec.template.containers[].restartPolicy = "OnFailure"`, then the Pod stays
on the node, but the Container is re-run.  Therefore, your program needs to handle the the case when it is
restarted locally, or else specify `.spec.template.containers[].restartPolicy = "Never"`.
See [pods-states](pod-states.md) for more information on `restartPolicy`.

An entire Pod can also fail, for a number of reasons, such as when the pod is kicked off the node
(node is upgraded, rebooted, deleted, etc.), or if a container of the Pod fails and the
`.spec.template.containers[].restartPolicy = "Never"`.  When a Pod fails, then the Job controller
starts a new Pod.  Therefore, your program needs to handle the case when it is restarted in a new
pod.  In particular, it needs to handle temporary files, locks, incomplete output and the like
caused by previous runs.

Note that even if you specify `.spec.parallelism = 1` and `.spec.completions = 1` and
`.spec.template.containers[].restartPolicy = "Never"`, the same program may
sometimes be started twice.

If you do specify `.spec.parallelism` and `.spec.completions` both greater than 1, then there may be
multiple pods running at once.  Therefore, your pods must also be tolerant of concurrency.

## Job Patterns

The Job object can be used to support reliable parallel execution of Pods.  The Job object is not
designed to support closely-communicating parallel processes, as commonly found in scientific
computing.  It does support parallel processing of a set of independent but related *work items*.
These might be emails to be sent, frames to be rendered, files to be transcoded, ranges of keys in a
NoSQL database to scan, and so on.

In a complex system, there may be multiple different sets of work items.  Here we are just
considering one set of work items that the user wants to manage together &mdash; a *batch job*.

There are several different patterns for parallel computation, each with strengths and weaknesses.
The tradeoffs are:

- One Job object for each work item, vs a single Job object for all work items.  The latter is
  better for large numbers of work items.  The former creates some overhead for the user and for the
  system to manage large numbers of Job objects.  Also, with the latter, the resource usage of the job
  (number of concurrently running pods) can be easily adjusted using the `kubectl scale` command.
- Number of pods created equals number of work items, vs each pod can process multiple work items.
  The former typically requires less modification to existing code and containers.  The latter
  is better for large numbers of work items, for similar reasons to the previous bullet.
- Several approaches use a work queue.  This requires running a queue service,
  and modifications to the existing program or container to make it use the work queue.
  Other approaches are easier to adapt to an existing containerised application.


The tradeoffs are summarized here, with columns 2 to 4 corresponding to the above tradeoffs.
The pattern names are also links to examples and more detailed description.

|                            Pattern                                         | Single Job object | Fewer pods than work items? | Use app unmodified? |  Works in Kube 1.1? |
| -------------------------------------------------------------------------- |:-----------------:|:---------------------------:|:-------------------:|:-------------------:|
| [Job Template Expansion](../../examples/job/expansions/README.md)          |                   |                             |          ✓          |          ✓          |
| [Queue with Pod Per Work Item](../../examples/job/work-queue-1/README.md)  |         ✓         |                             |      sometimes      |          ✓          |
| [Queue with Variable Pod Count](../../examples/job/work-queue-2/README.md) |                   |         ✓         |             ✓               |                     |          ✓          |
| Single Job with Static Work Assignment                                     |         ✓         |                             |          ✓          |                     |

When you specify completions with `.spec.completions`, each Pod created by the Job controller
has an identical [`spec`](../devel/api-conventions.md#spec-and-status).  This means that
all pods will have the same command line and the same
image, the same volumes, and (almost) the same environment variables.  These patterns
are different ways to arrange for pods to work on different things.

This table shows the required settings for `.spec.parallelism` and `.spec.completions` for each of the patterns.
Here, `W` is the number of work items.

|                             Pattern                                        | `.spec.completions` |  `.spec.parallelism` |
| -------------------------------------------------------------------------- |:-------------------:|:--------------------:|
| [Job Template Expansion](../../examples/job/expansions/README.md)          |          1          |     should be 1      |
| [Queue with Pod Per Work Item](../../examples/job/work-queue-1/README.md)  |          W          |        any           |
| [Queue with Variable Pod Count](../../examples/job/work-queue-2/README.md) |          1          |        any           |
| Single Job with Static Work Assignment                                     |          W          |        any           |



## Alternatives

### Bare Pods

When the node that a pod is running on reboots or fails, the pod is terminated
and will not be restarted.  However, a Job will create new pods to replace terminated ones.
For this reason, we recommend that you use a job rather than a bare pod, even if your application
requires only a single pod.

### Replication Controller

Jobs are complementary to [Replication Controllers](replication-controller.md).
A Replication Controller manages pods which are not expected to terminate (e.g. web servers), and a Job
manages pods that are expected to terminate (e.g. batch jobs).

As discussed in [life of a pod](pod-states.md), `Job` is *only* appropriate for pods with
`RestartPolicy` equal to `OnFailure` or `Never`.  (Note: If `RestartPolicy` is not set, the default
value is `Always`.)

### Single Job starts Controller Pod

Another pattern is for a single Job to create a pod which then creates other pods, acting as a sort
of custom controller for those pods.  This allows the most flexibility, but may be somewhat
complicated to get started with and offers less integration with Kubernetes.

One example of this pattern would be a Job which starts a Pod which runs a script that in turn
starts a Spark master controller (see [spark example](../../examples/spark/README.md)), runs a spark
driver, and then cleans up.

An advantage of this approach is that the overall process gets the completion guarantee of a Job
object, but complete control over what pods are created and how work is assigned to them.

## Caveats

Job objects are in the [`extensions` API Group](../api.md#api-groups).

Job objects have [API version `v1beta1`](../api.md#api-versioning).  Beta objects may
undergo changes to their schema and/or semantics in future software releases, but
similar functionality will be supported.

## Future work

Support for creating Jobs at specified times/dates (i.e. cron) is expected in the next minor
release.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/jobs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
