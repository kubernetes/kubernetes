<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes architecture

A running Kubernetes cluster contains node agents (`kubelet`) and master
components (APIs, scheduler, etc), on top of a distributed storage solution.
This diagram shows our desired eventual state, though we're still working on a
few things, like making `kubelet` itself (all our components, really) run within
containers, and making the scheduler 100% pluggable.

![Architecture Diagram](architecture.png?raw=true "Architecture overview")

## The Kubernetes Node

When looking at the architecture of the system, we'll break it down to services
that run on the worker node and services that compose the cluster-level control
plane.

The Kubernetes node has the services necessary to run application containers and
be managed from the master systems.

Each node runs Docker, of course.  Docker takes care of the details of
downloading images and running containers.

### `kubelet`

The `kubelet` manages [pods](../user-guide/pods.md) and their containers, their
images, their volumes, etc.

### `kube-proxy`

Each node also runs a simple network proxy and load balancer (see the
[services FAQ](https://github.com/kubernetes/kubernetes/wiki/Services-FAQ) for
more details). This reflects `services` (see
[the services  doc](../user-guide/services.md) for more details) as defined in
the Kubernetes API on each node and can do simple TCP and UDP stream forwarding
(round robin) across a set of backends.

Service endpoints are currently found via [DNS](../admin/dns.md) or through
environment variables (both
[Docker-links-compatible](https://docs.docker.com/userguide/dockerlinks/) and
Kubernetes `{FOO}_SERVICE_HOST` and `{FOO}_SERVICE_PORT` variables are
supported). These variables resolve to ports managed by the service proxy.

## The Kubernetes Control Plane

The Kubernetes control plane is split into a set of components. Currently they
all run on a single _master_ node, but that is expected to change soon in order
to support high-availability clusters. These components work together to provide
a unified view of the cluster.

### `etcd`

All persistent master state is stored in an instance of `etcd`. This provides a
great way to store configuration data reliably. With `watch` support,
coordinating components can be notified very quickly of changes.

### Kubernetes API Server

The apiserver serves up the [Kubernetes API](../api.md). It is intended to be a
CRUD-y server, with most/all business logic implemented in separate components
or in plug-ins. It mainly processes REST operations, validates them, and updates
the corresponding objects in `etcd` (and eventually other stores).

### Scheduler

The scheduler binds unscheduled pods to nodes via the `/binding` API. The
scheduler is pluggable, and we expect to support multiple cluster schedulers and
even user-provided schedulers in the future.

### Kubernetes Controller Manager Server

All other cluster-level functions are currently performed by the Controller
Manager. For instance, `Endpoints` objects are created and updated by the
endpoints controller, and nodes are discovered, managed, and monitored by the
node controller. These could eventually be split into separate components to
make them independently pluggable.

The [`replicationcontroller`](../user-guide/replication-controller.md) is a
mechanism that is layered on top of the simple [`pod`](../user-guide/pods.md)
API. We eventually plan to port it to a generic plug-in mechanism, once one is
implemented.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/architecture.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
