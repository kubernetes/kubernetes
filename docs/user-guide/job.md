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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/job.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Example of a Job

The following directory contains a sample [job](../../docs/proposals/job.md) you can run on top of Kubernetes, with information about the expected results.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](prereqs.md) for installation instructions for your platform.

### Simple job

In general the definition of a job does not differ that much from a [Replication Controller](replication-controller.md). The main difference being:

* pod's restart policy (`OnFailure` - will restart failed pods, `Never` - won't),
* completions (how many cumulative runs of a pod will exists), similar to RC's replicas,
* parallelism (how many active pods runs at any given time).

Define job:

```console
$ kubectl create -f ./job.yaml
```

Where job.yaml contains something like this:

<!-- BEGIN MUNGE: EXAMPLE job.yaml -->

```yaml
apiVersion: v1
kind: Job
metadata:
  name: test-job
spec:
  parallelism: 2
  completions: 5
  selector:
    job: test
  template:
    metadata:
      labels:
        job: test
    spec:
      restartPolicy: OnFailure
      containers:
      - name: basic-pod
        image: "centos:centos7"
        command: ['bash', '-c', 'echo ok']
```

[Download example](job.yaml?raw=true)
<!-- END MUNGE: EXAMPLE job.yaml -->

You should now see at most 2 (`.spec.parallelism`) pods running this job until total amount of successful executions (`.spec.template.spec.restartPolicy`) will reach 5 (`.spec.completions`). You can verify this with:

```console
$ kubectl describe jobs/test-job
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/job.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
