<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
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

# CRI: Log management for container stdout/stderr streams


## Goals and non-goals

Container Runtime Interface (CRI) is an ongoing project to allow container
runtimes to integrate with kubernetes via a newly-defined API. The goal of this
proposal is to define how container's *stdout/stderr* log streams should be
handled in CRI.

The explicit non-goal is to define how (non-stdout/stderr) application logs
should be handled. Collecting and managing arbitrary application logs is a
long-standing issue [1] in kubernetes and is worth a proposal of its own. Even
though this proposal does not touch upon these logs, the direction of
this proposal is aligned with one of the most-discussed solutions, logging
volumes [1], for general logging management.

*In this proposal, “logs” refer to the stdout/stderr streams of the
containers, unless specified otherwise.*

Previous CRI logging issues:
 - Tracking issue: https://github.com/kubernetes/kubernetes/issues/30709
 - Proposal (by @tmrtfs): https://github.com/kubernetes/kubernetes/pull/33111

The scope of this proposal is narrower than the #33111 proposal, and hopefully
this will encourage a more focused discussion.


## Background

Below is a brief overview of logging in kubernetes with docker, which is the
only container runtime with fully functional integration today.

**Log lifecycle and management**

Docker supports various logging drivers (e.g., syslog, journal, and json-file),
and allows users to configure the driver by passing flags to the docker daemon
at startup. Kubernetes defaults to the "json-file" logging driver, in which
docker writes the stdout/stderr streams to a file in the json format as shown
below.

```
{“log”: “The actual log line”, “stream”: “stderr”, “time”: “2016-10-05T00:00:30.082640485Z”}
```

Docker deletes the log files when the container is removed, and a cron-job (or
systemd timer-based job) on the node is responsible to rotate the logs (using
`logrotate`). To preserve the logs for introspection and debuggability, kubelet
keeps the terminated container until the pod object has been deleted from the
apiserver.

**Container log retrieval**

The kubernetes CLI tool, kubectl, allows users to access the container logs
using [`kubectl logs`]
(http://kubernetes.io/docs/user-guide/kubectl/kubectl_logs/) command.
`kubectl logs` supports flags such as `--since` that requires understanding of
the format and the metadata (i.e., timestamps) of the logs. In the current
implementation, kubelet calls `docker logs` with parameters to return the log
content. As of now, docker only supports `log` operations for the “journal” and
“json-file” drivers [2]. In other words, *the support of `kubectl logs` is not
universal in all kuernetes deployments*.

**Cluster logging support**

In a production cluster, logs are usually collected, aggregated, and shipped to
a remote store where advanced analysis/search/archiving functions are
supported. In kubernetes, the default cluster-addons includes a per-node log
collection daemon, `fluentd`. To facilitate the log collection, kubelet creates
symbolic links to all the docker containers logs under `/var/log/containers`
with pod and container metadata embedded in the filename.

```
/var/log/containers/<pod_name>_<pod_namespace>_<container_name>-<container_id>.log`
```

The fluentd daemon watches the `/var/log/containers/` directory and extract the
metadata associated with the log from the path. Note that this integration
requires kubelet to know where the container runtime stores the logs, and will
not be directly applicable to CRI.


## Requirements

   1. **Provide ways for CRI-compliant runtimes to support all existing logging
        features, i.e., `kubectl logs`.**

   2. **Allow kubelet to manage the lifecycle of the logs to pave the way for
        better disk management in the future.** This implies that the lifecycle
        of containers and their logs need to be decoupled.

   3. **Allow log collectors to easily integrate with Kubernetes across
        different container runtimes while preserving efficient storage and
        retrieval.**

Requirement (1) provides opportunities for runtimes to continue support
`kubectl logs --since` and related features. Note that even though such
features are only supported today for a limited set of log drivers, this is an
important usability tool for a fresh, basic kubernetes cluster, and should not
be overlooked. Requirement (2) stems from the fact that disk is managed by
kubelet as a node-level resource (not per-pod) today, hence it is difficult to
delegate to the runtime by enforcing per-pod disk quota policy. In addition,
container disk quota is not well supported yet, and such limitation may not
even be well-perceived by users. Requirement (1) is crucial to the kubernetes'
extensibility and usability across all deployments.

## Proposed solution

This proposal intends to satisfy the requirements by

  1. Enforce where the container logs should be stored on the host
     filesystem. Both kubelet and the log collector can interact with
     the log files directly.

  2. Ask the runtime to decorate the logs in a format that kubelet understands.

**Log directories and structures**

Kubelet will be configured with a root directory (e.g., `/var/log/pods` or
`/var/lib/kubelet/logs/) to store all container logs. Below is an example of a
path to the log of a container in a pod.

```
/var/log/pods/<podUID>/<containerName>_<instance#>.log
```

In CRI, this is implemented by setting the pod-level log directory when
creating the pod sandbox, and passing the relative container log path
when creating a container.

```
PodSandboxConfig.LogDirectory: /var/log/pods/<podUID>/
ContainerConfig.LogPath: <containerName>_<instance#>.log
```

Because kubelet determines where the logs are stores and can access them
directly, this meets requirement (1). As for requirement (2), the log collector
can easily extract basic pod metadata (e.g., pod UID, container name) from
the paths, and watch the directly for any changes. In the future, we can
extend this by maintaining a metada file in the pod directory.

**Log format**

The runtime should decorate each log entry with a RFC 3339Nano timestamp
prefix, the stream type (i.e., "stdout" or "stderr"), and ends with a newline.

```
2016-10-06T00:17:09.669794202Z stdout The content of the log entry 1
2016-10-06T00:17:10.113242941Z stderr The content of the log entry 2
```

With the knowledge, kubelet can parses the logs and serve them for `kubectl
logs` requests. This meets requirement (3). Note that the format is defined
deliberately simple to provide only information necessary to serve the requests.
We do not intend for kubelet to host various logging plugins. It is also worth
mentioning again that the scope of this proposal is restricted to stdout/stderr
streams of the container, and we impose no restriction to the logging format of
arbitrary container logs.

**Who should rotate the logs?**

We assume that a separate task (e.g., cron job) will be configured on the node
to rotate the logs periodically, similar to today’s implementation.

We do not rule out the possibility of letting kubelet or a per-node daemon
(`DaemonSet`) to take up the responsibility, or even declare rotation policy
in the kubernetes API as part of the `PodSpec`, but it is beyond the scope of
the this proposal.

**What about non-supported log formats?**

If a runtime chooses to store logs in non-supported formats, it essentially
opts out of `kubectl logs` features, which is backed by kubelet today. It is
assumed that the user can rely on the advanced, cluster logging infrastructure
to examine the logs.

It is also possible that in the future, `kubectl logs` can contact the cluster
logging infrastructure directly to serve logs [1a]. Note that this does not
eliminate the need to store the logs on the node locally for reliability.


**How can existing runtimes (docker/rkt) comply to the logging requirements?**

In the short term, the ongoing docker-CRI integration [3] will support the
proposed solution only partially by (1) creating symbolic links for kubelet
to access, but not manage the logs, and (2) add support for json format in
kubelet. A more sophisticated solution that either involves using a custom
plugin or launching a separate process to copy and decorate the log will be
considered as a mid-term solution.

For rkt, implementation will rely on providing external file-descriptors for
stdout/stderr to applications via systemd [4]. Those streams are currently
managed by a journald sidecar, which collects stream outputs and store them
in the journal file of the pod. This will replaced by a custom sidecar which
can produce logs in the format expected by this specification and can handle
clients attaching as well.

## Alternatives

There are ad-hoc solutions/discussions that addresses one or two of the
requirements, but no comprehensive solution for CRI specifically has been
proposed so far (with the excpetion of @tmrtfs's proposal
[#33111](https://github.com/kubernetes/kubernetes/pull/33111), which has a much
wider scope). It has come up in discussions that kubelet can delegate all the
logging management to the runtime to allow maximum flexibility. However, it is
difficult for this approach to meet either requirement (1) or (2), without
defining complex logging API.

There are also possibilities to implement the current proposal by imposing the
log file paths, while leveraging the runtime to access and/or manage logs. This
requires the runtime to expose knobs in CRI to retrieve, remove, and examine
the disk usage of logs. The upside of this approach is that kubelet needs not
mandate the logging format, assuming runtime already includes plugins for
various logging formats. Unfortunately, this is not true for existing runtimes
such as docker, which supports log retrieval only for a very limited number of
log drivers [2]. On the other hand, the downside is that we would be enforcing
more requirements on the runtime through log storage location on the host, and
a potentially premature logging API that may change as the disk management
evolves.

## References

[1] Log management issues:
 - a. https://github.com/kubernetes/kubernetes/issues/17183
 - b. https://github.com/kubernetes/kubernetes/issues/24677
 - c. https://github.com/kubernetes/kubernetes/pull/13010

[2] Docker logging drivers:
 - https://docs.docker.com/engine/admin/logging/overview/

[3] Docker CRI integration:
 - https://github.com/kubernetes/kubernetes/issues/31459

[4] rkt support: https://github.com/systemd/systemd/pull/4179



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-cri-logging.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
