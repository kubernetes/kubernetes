# Logging in Container Runtime Interface

The following proposal summarizes the current problems encountered during
cluster log aggregation in Kubernetes and proposes a solution to the problem on
the container runtime level.

## Motivation

Logging is an essential part of cluster deployments, currently the cluster
logging is facilitated via adhoc solutions and plugins. Use of different
runtimes and heterogenity of logging format ecosystem makes it difficult to have
a clean first-class solution to logging.

In order to simplify the deployments and reduce the developer workload, it is
imperative to provide a sane way to expose application logs and context
metadata.

## Solution

The proposed solution is a volume-based approach and a loose logging format that
will be specified using the container runtime interface to provide a coherent
logging API.

Kubernetes will have logging volumes for containers that will be used by
runtimes. Each container will be assigned an append-only file on the logging
volume of the pod. The logging volume lifecycle will be managed by Kubernetes,
namely rotation and truncation, according to the logging policy specified by the
user.

It will be the container runtime's responsibility to redirect the logs of a container
to the specified location, be it stdout/stderr logs or log files themselves.

## Log Delimiting

We cannot hope to dictate the logging format that will be used by the consumers.
However, in order to perform certain operations such as rotation, Kubernetes
needs to be aware of the boundaries of log units, which are logs that are
meaningless when broken up. So, Kubernetes will ask runtimes to use a delimiter
for the log lines. For the purposes of this proposal a unit of logging is called
an **Event**.

When rotation/truncation is being performed, only X events will be used, which
is the biggest number of events that fit into the specified maximum log size.

With the exception of the afore-mentioned delimiter, Kubernetes will treat logs
as opaque.

## Log Redirection

Containers can log to stdout/stderr or the filesystem, and it is the runtime's
responsibility to redirect the container logs to the path specified in the
logging volume.

## Log Volumes

Logs will be redirected to a logging volume by the container runtime.

The specification won't have any indications as to which runtime is being used.

Each runtime will have implementations conforming to the proposed logging solution.

An example log volume specification adapted from the previous `logsDir` effort is as follows:

```go
spec:
    volumes:
        - name mylogs
          emptyDir: {}
    containers:
        - name: foo
          volumeMounts:
              name: mylogs
              path: /var/log/
              policy:
                  logs:
                      subDir: foo
                      rotate: Daily
                      annotations:
                          "fluentd-config": "actual fluentd configuration" 
        - name: bar
          volumeMounts:
              name: mylogs
              path: /var/log/
              policy:
                  logs:
                      subDir: bar
                      rotate: Hourly
                      annotations:
                          "fluentd-config": "actual fluentd configuration" 
```

### Logging Policies

Kubernetes will perform log rotation and trucation according to the specified
logging policy using the loose logging format.

The responsibility of container runtimes will be to redirect container logs and
send a delimiter with the logs to separate logs into logging events (e.g. line
endings).

## Log API

The proposed mechanism includes a first-class API endpoint for reaching the
container and pod logs `/logs/<pod>/<container>`.

Logging aggregators and daemons will consume the logs plus the context metadata of those logs.
The logs will be exposed by Kubernetes and populated by container runtimes.
