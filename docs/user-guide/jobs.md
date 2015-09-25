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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/jobs.md).

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
    - [Multiple Completions](#multiple-completions)
    - [Parallelism](#parallelism)
  - [Handling Pod and Container Failures](#handling-pod-and-container-failures)
  - [Alternatives to Job](#alternatives-to-job)
    - [Bare Pods](#bare-pods)
    - [Replication Controller](#replication-controller)
  - [Caveats](#caveats)
  - [Future work](#future-work)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _job_?

A _job_ creates one or more pods and ensures that a specified number of them successfully terminate.
As pods successfully complete, the _job_ tracks the successful completions.  When a specified number
of successful completions is reached, the job itself is complete.  Deleting a Job will cleanup the
pods it created.

A simple case is to create 1 Job object in order to reliably run one Pod to completion.
A Job can also be used to run multiple pods in parallel.

## Running an example Job

Here is an example Job config.  It computes π to 2000 places and prints it out.
It takes around 10s to complete.
<!-- BEGIN MUNGE: EXAMPLE job.yaml -->

```yaml
apiVersion: experimental/v1alpha1
kind: Job
metadata:
  name: pi
spec:
  selector:
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
jobs/pi
```

Check on the status of the job using this command:

```console
$ kubectl describe jobs/pi
Name:		pi
Namespace:	default
Image(s):	perl
Selector:	app=pi
Parallelism:	2
Completions:	1
Labels:		<none>
Pods Statuses:	1 Running / 0 Succeeded / 0 Failed
Events:
  FirstSeen	LastSeen	Count	From	SubobjectPath	Reason			Message
  ─────────	────────	─────	────	─────────────	──────			───────
  1m		1m		1	{job }			SuccessfulCreate	Created pod: pi-z548a

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
$ kubectl logs pi-aiw0a
3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989380952572010654858632788659361533818279682303019520353018529689957736225994138912497217752834791315155748572424541506959508295331168617278558890750983817546374649393192550604009277016711390098488240128583616035637076601047101819429555961989467678374494482553797747268471040475346462080466842590694912933136770289891521047521620569660240580381501935112533824300355876402474964732639141992726042699227967823547816360093417216412199245863150302861829745557067498385054945885869269956909272107975093029553211653449872027559602364806654991198818347977535663698074265425278625518184175746728909777727938000816470600161452491921732172147723501414419735685481613611573525521334757418494684385233239073941433345477624168625189835694855620992192221842725502542568876717904946016534668049886272327917860857843838279679766814541009538837863609506800642251252051173929848960841284886269456042419652850222106611863067442786220391949450471237137869609563643719172874677646575739624138908658326459958133904780275901
```

## Writing a Job Spec

As with all other Kubernetes config, a Job needs `apiVersion`, `kind`, and `metadata` fields.  For
general information about working with config files, see [here](simple-yaml.md),
[here](configuring-containers.md), and [here](working-with-resources.md).

A Job also needs a [`.spec` section](../devel/api-conventions.md#spec-and-status).

### Pod Template

The `.spec.template` is the only required field of the `.spec`.

The `.spec.template` is a [pod template](replication-controller.md#pod-template).  It has exactly
the same schema as a [pod](pods.md), except it is nested and does not have an `apiVersion` or
`kind`.

In addition to required fields for a Pod, a pod template in a job must specify appropriate
lables (see [pod selector](#pod-selector) and an appropriate restart policy.

Only a [`RestartPolicy`](pod-states.md) equal to `Never` or `OnFailure` are allowed.

### Pod Selector

The `.spec.selector` field is a pod selector.  It works the same as the `.spec.selector` of
a [ReplicationController](replication-controller.md).

If specified, the `.spec.template.metadata.labels` must be equal to the `.spec.selector`, or it will
be rejected by the API.  If `.spec.selector` is unspecified, it will be defaulted to
`.spec.template.metadata.labels`.

Also you should not normally create any pods whose labels match this selector, either directly,
via another Job, or via another controller such as ReplicationController.  Otherwise, the Job will
think that those pods were created by it.  Kubernetes will not stop you from doing this.

### Multiple Completions

By default, a Job is complete when one Pod runs to successful completion.  You can also specify that
this needs to happen multiple times by specifying `.spec.completions` with a value greater than 1.
When multiple completions are requested, each Pod created by the Job controller has an identical
[`spec`](../devel/api-conventions.md#spec-and-status).  In particular, all pods will have
the same command line and the same image, the same volumes, and mostly the same environment
variables.  It is up to the user to arrange for the pods to do work on different things.  For
example, the pods might all access a shared work queue service to acquire work units.

To create multiple pods which are similar, but have slightly different arguments, environment
variables or images, use multiple Jobs.

### Parallelism

You can suggest how many pods should run concurrently by setting `.spec.parallelism` to the number
of pods you would like to have running concurrently.  This number is a suggestion. The number
running concurrently may be lower or higher for a variety of reasons.  For example, it may be lower
if the number of remaining completions is less, or as the controller is ramping up, or if it is
throttling the job due to excessive failures.  It may be higher for example if a pod is gracefully
shutdown, and the replacement starts early.

If you do not specify `.spec.parallelism`, then it defaults to `.spec.completions`.

## Handling Pod and Container Failures

A Container in a Pod may fail for a number of reasons, such as because the process in it exited with
a non-zero exit code, or the Container was killed for exceeding a memory limit, etc.  If this
happens, and the `.spec.template.containers[].restartPolicy = "OnFailure"`, then the Pod stays
on the node, but the Container is re-run.  Therefore, your program needs to handle the the case when it is
restarted locally, or else specify `.spec.template.containers[].restartPolicy = "Never"`.
See [pods-states](pod-states.md) for more information on `restartPolicy`.

An entire Pod can also fail, for a number of reasons, such as when the pod is kicked off the node
(node is upgraded, rebooted, delelted, etc.), or if a container of the Pod fails and the
`.spec.template.containers[].restartPolicy = "Never"`.  When a Pod fails, then the Job controller
starts a new Pod.  Therefore, your program needs to handle the case when it is restarted in a new
pod.  In particular, it needs to handle temporary files, locks, incomplete output and the like
caused by previous runs.

Note that even if you specify `.spec.parallelism = 1` and `.spec.completions = 1` and
`.spec.template.containers[].restartPolicy = "Never"`, the same program may
sometimes be started twice.

If you do specify `.spec.parallelism` and `.spec.completions` both greater than 1, then there may be
multiple pods running at once.  Therefore, your pods must also be tolerant of concurrency.

## Alternatives to Job

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

## Caveats

Job is part of the experimental API group, so it is not subject to the same compatibility
guarantees as objects in the main API.  It may not be enabled.  Enable by setting
`--runtime-config=experimental/v1alpha1` on the apiserver.

## Future work

Support for creating Jobs at specified times/dates (i.e. cron) is expected in the next minor
release.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/jobs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
