# Kubernetes-Mesos Scheduler

Kubernetes on Mesos does not use the upstream scheduler binary, but replaces it
with its own Mesos framework scheduler. The following gives an overview of
the differences.

## Labels and Mesos Agent Attributes

The scheduler of Kubernetes-Mesos takes [labels][1] into account: it matches
specified labels in pod specs with defined labels of nodes.

In addition to user defined labels, [attributes of Mesos agents][2] are converted
into node labels by the scheduler, following the pattern

```yaml
k8s.mesosphere.io/attribute-<name>: value
```

As an example, a Mesos agent attribute of `generation:2015` will result in the node label

```yaml
k8s.mesosphere.io/attribute-generation: 2015
```

and can be used to schedule pods onto nodes which are of generation 2015.

**Note:** Node labels prefixed by `k8s.mesosphere.io` are managed by
Kubernetes-Mesos and should not be modified manually by the user or admin. For
example, the Kubernetes-Mesos executor manages `k8s.mesosphere.io/attribute`
labels and will auto-detect and update modified attributes when the mesos-slave
is restarted.

## Resource Roles

A Mesos cluster can be statically partitioned using [resources roles][2]. Each
resource is assigned such a role (`*` is the default role, if none is explicitly
assigned in the mesos-slave command line). The Mesos master will send offers to
frameworks for `*` resources and – optionally – for one extra role that a
framework is assigned to. Right now only one such extra role for a framework is
supported.

### Configuring Roles for the Scheduler

Every Mesos framework scheduler can choose among the offered `*` resources and
those of the extra role. The Kubernetes-Mesos scheduler supports this by setting
the framework roles in the scheduler command line, e.g.

```bash
$ km scheduler ... --mesos-roles="*,role1" ...
```

This will tell the Kubernetes-Mesos scheduler to default to using `*` resources
if a pod is not specially assigned to another role. Moreover, the extra role
`role1` is allowed, i.e. the Mesos master will send resources or role `role1`
to the Kubernetes scheduler.

Note the following restrictions and possibilities:
- Due to the restrictions of Mesos, only one extra role may be provided on the
  command line.
- It is allowed to only pass an extra role without the `*`, e.g. `--mesos-roles=role1`.
  This means that no `*` resources should be considered by the scheduler at all.
- It is allowed to pass the extra role first, e.g. `--mesos-roles=role1,*`.
  This means that `role1` is the default role for pods without special role
  assignment (see below). But `*` resources would be considered for pods with a special `*`
  assignment.

### Specifying Roles for Pods

By default a pod is scheduled using resources of the role which comes first in
the list of scheduler roles.

A pod can opt-out of this default behaviour using the `k8s.mesosphere.io/roles`
label:

```yaml
k8s.mesosphere.io/roles: role1,role2,role3
```

The format is a comma separated list of allowed resource roles. The scheduler
will try to schedule the pod with `role1` resources first, using `role2`
resources if the former are not available and finally falling back to `role3`
resources.

The `*` role may be specified as well in this list.

**Note:** An empty list will mean that no resource roles are allowed which is
equivalent to a pod which is unschedulable.

For example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend
  labels:
    k8s.mesosphere.io/roles: *,prod,test,dev
  namespace: prod
spec:
  ...
```

This `prod/backend` pod will be scheduled using resources from all four roles,
preferably using `*` resources, followed by `prod`, `test` and `dev`. If none
of those for roles provides enough resources, the scheduling fails.

**Note:** The scheduler will also allow to mix different roles in the following
sense: if a node provides `cpu` resources for the `*` role, but `mem` resources
only for the `prod` role, the upper pod will be schedule using `cpu(*)` and
`mem(prod)` resources.

**Note:** The scheduler might also mix within one resource type, i.e. it will
use as many `cpu`s of the `*` role as possible. If a pod requires even more
`cpu` resources (defined using the `pod.spec.resources.limits` property) for successful
scheduling, the scheduler will add resources from the `prod`, `test` and `dev`
roles, in this order until the pod resource requirements are satisfied. E.g. a
pod might be scheduled with 0.5 `cpu(*)`, 1.5 `cpu(prod)` and 1 `cpu(test)`
resources plus e.g. 2 GB `mem(prod)` resources.

## Tuning

The scheduler configuration can be fine-tuned using an ini-style configuration file.
The filename is passed via `--scheduler-config` to the `km scheduler` command.

Be warned though that some them are pretty low-level and one has to know the inner
workings of k8sm to find sensible values. Moreover, these settings may change or
even disappear from version to version without further notice.

The following settings are the default:

```
[scheduler]
; duration an offer is viable, prior to being expired
offer-ttl = 5s

; duration an expired offer lingers in history
offer-linger-ttl = 2m

<<<<<<< HEAD
; duration between offer listener notifications
listener-delay = 1s

; size of the pod updates channel
updates-backlog = 2048

; interval we update the frameworkId stored in etcd
framework-id-refresh-interval = 30s

; wait this amount of time after initial registration before attempting
; implicit reconciliation
initial-implicit-reconciliation-delay = 15s

; interval in between internal task status checks/updates
explicit-reconciliation-max-backoff = 2m

; waiting period after attempting to cancel an ongoing reconciliation
explicit-reconciliation-abort-timeout = 30s

initial-pod-backoff = 1s
max-pod-backoff = 60s
http-handler-timeout = 10s
http-bind-interval = 5s
```

## Low-Level Scheduler Architecture

![Scheduler Structure](scheduler.png)

[1]: ../../../docs/user-guide/labels.md
[2]: http://mesos.apache.org/documentation/attributes-resources/

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/scheduler.md?pixel)]()
