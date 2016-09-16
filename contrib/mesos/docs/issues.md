## Known Issues

This page identifies significant known issues with the Kubernetes-Mesos distribution.

* [General Known Issues](#general-known-issues)
* [DCOS Package Known Issues](#dcos-package-known-issues), in addendum to the above.

## General Known Issues

These known issues apply to all builds of Kubernetes-Mesos.

### Upgrades

Upgrading your Kubernetes-Mesos cluster is currently unsupported.
One known problem exists with respect to expressing executor (kubelet and kube-proxy) process configuration via command line flags.
It is **strongly** recommended that all of the Kubernetes-Mesos executors are destroyed before upgrading the Kubernetes-Mesos scheduler component:
- destroy all daemon controllers running in the cluster, across all namespaces
- destroy all replication controllers running in the cluster, across all namespaces
- destroy all pods running in the cluster, across all namespaces
- invoke the "kamikaze" debug endpoint on the scheduler (e.g. `curl http://10.2.0.5:10251/debug/actions/kamikaze`) to terminate all executors

Not following the above steps prior to upgrading the scheduler can result in a cluster wherein pods will never again be scheduled upon one or more nodes.
This issue is being tracked here: https://github.com/mesosphere/kubernetes-mesos/issues/572.

### Netfilter Connection Tracking

The scheduler offers flags to tweak connection tracking for kube-proxy instances that are launched on slave nodes:

- conntrack-max (do **NOT** set this to a non-zero value if the Mesos slave process is running in a non-root network namespace)
- conntrack-tcp-timeout-established

By default both of these are set to 0 when running Kubernetes-Mesos.
Setting either of these flags to non-zero values may impact connection tracking for the entire slave.

### Port Specifications

In order for pods (replicated, or otherwise) to be scheduled on the cluster, it is strongly recommended that:
* `pod.spec.containers[x].ports[y].hostPort` be left unspecified (or zero), or else;
* `pod.spec.containers[x].ports[y].hostPort` exists in the range of `ports` resources declared on Mesos slaves
  - double-check the resource declarations for your Mesos slaves, the default for `ports` is typically `[31000-32000]`

Mesos slave host `ports` are resources that are managed by the Mesos resource/offers ecosystem; slave host ports are consumed by launched tasks.
Kubernetes pod container specifications identify two types of ports, "container ports" and "host ports":
- container ports are allocated from the network namespace of the pod, which is independent from that of the host, whereas;
- host ports are allocated from the network namespace of the host.

**Notable on Kubernetes-Mesos**
- Mesos slaves must be configured to offer host `ports` resources in order for pods to use them. Most Mesos package distributions, by default, configure a `ports` resource range for each slave.
- The scheduler recognizes the declared *host ports* of each container in a pod/task and for each such host port, attempts to allocate it from the offered port resources listed in Mesos offers.
- If no host port is declared for a given port spec, then the scheduler may map that port spec's container port to any host port from the offered ports ranges.
- Any *host ports* explicitly declared in the pod container specification must fall within that range of `ports` offered by slaves in the cluster.
  Ports declared outside that range (other than zero) will never match resource offers received by the scheduler, and so pod specifications that declare such ports will never be executed as tasks on the cluster.
- A missing pod container host port declaration or a host port set to zero will, by default, result in the allocation of a host port from a resource offer.
- If a pod is the target of a Kubernetes service selector then the related target container ports must be declared in the pod spec.
- In vanilla Kubernetes, host ports with the value zero are ignored.
  To obtain the same behavior with the Kubernetes-Mesos scheduler pods must be assigned a label of `k8s.mesosphere.io/portMapping` with the value `fixed`
  (see [#527](https://github.com/mesosphere/kubernetes-mesos/issues/527)).

### Pods

#### Pod Updates

Once a task has been launched for a given pod, Kubernetes-Mesos is blind to any updates applied to the pod state (other than for forced, or graceful deletion).

#### Pod Placement

The initial plan was to implement pod placement (aka scheduling "constraints") using rules similar to those found in Marathon.
Upon further consideration it has been decided that a greater alignment between the stock Kubernetes scheduler and Kubernetes-Mesos scheduler would benefit both projects, as well as end-users.
Currently there is limited support for pod placement using the Kubernetes-Mesos [scheduler](scheduler.md).
This issue is being tracked here: https://github.com/mesosphere/kubernetes-mesos/issues/338

**Note:** An upcoming changeset will update the scheduler with initial support for multiple Mesos roles
(see [#482](https://github.com/mesosphere/kubernetes-mesos/issues/482)).

#### Static Pods

Static pods are supported by the scheduler.
The path to a directory containing pod definitions can be set via the `--static-pods-config` flag.
Static pods are subject to the following restrictions:

- Static pods *are read only once* by the scheduler on startup.
  Only newly started executor will get the latest static pod specs from the defined static pod directory.

#### Orphan Pods

The default `executor_shutdown_grace_period` of a Mesos slave is 3 seconds.
When the executor is shut down it forcefully terminates the Docker containers that it manages.
However, if terminating the Docker containers takes longer than the `executor_shutdown_grace_period` then some containers may not get a termination signal at all.
A consequence of this is that some pod containers, previously managed by the framework's executor, will remain running on the slave indefinitely.

There are two work-arounds to this problem:
* Restart the framework and it should terminate the orphaned tasks.
* Adjust the value of `executor_shutdown_grace_period` to something greater than 3 seconds.

### Services

#### Port Specifications

In order for Endpoints (therefore, Services) to be fully operational, it is strongly recommended that:
- service ports explicitly define a `name`
- service ports explicitly define a `targetPort`

For example:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-master
  labels:
    app: redis
    role: master
    tier: backend
spec:
  ports:
    # the port that this service should serve on
  - port: 6379
    targetPort: 6379
    name: k8sm-works-best-with-a-name-here
  selector:
    app: redis
    role: master
    tier: backend
```

#### Endpoints

At the time of this writing both Kubernetes and Mesos are using IPv4 addressing, albeit under different assumptions.
Mesos clusters configured with Docker typically use default Docker networking, which is host-private.
Kubernetes clusters assume a custom Docker networking configuration that assigns a cluster-routable IPv4 address to each pod, meaning that a process running anywhere on a Kubernetes cluster can reach a pod running on the same cluster by using the pod's Docker-assigned IPv4 address.

Kubernetes service endpoints terminate, by default, at a backing pod's IPv4 address using the container-port selected for in the service specification (PodIP:ContainerPort).
This is problematic when default Docker networking has been configured, such as in the case of typical Mesos clusters, because a pod's host-private IPv4 address is not intended to be reachable outside of its host.

The Kubernetes-Mesos project has implemented a work-around:
  service endpoints are terminated at HostIP:HostPort, where the HostIP is the IP address of the Mesos slave and the HostPort is the host port declared in the pod container port specification.
Host ports that are not defined, or else defined as zero, will automatically be assigned a (host) port resource from a resource offer.

To disable the work-around and revert to vanilla Kubernetes service endpoint termination:

- execute the k8sm scheduler with `-host-port-endpoints=false`
- execute the k8sm controller-manager with `-host-port-endpoints=false`

Then the usual Kubernetes network assumptions must be fulfilled for Kubernetes to work with Mesos, i.e. each container must get a cluster-wide routable IP (compare [Kubernetes Networking documentation](../../../docs/design/networking.md#container-to-container)).

This workaround may be mitigated down the road by:
- Future support for IPv6 addressing in Docker and Kubernetes
- Native IP-per-container support via Mesos with a custom Kubernetes network plugin

### Scheduling

Statements in this section regarding the "scheduler" pertain specifically to the Kubernetes-Mesos scheduler, unless otherwise noted.

Some factors that influence when pods are scheduled by k8s-mesos:
- availability of a resource offer that "fits" the pod (mesos master/slave);
- scheduler *backoff* (to avoid busy-looping) during pod scheduling (k8s-mesos scheduler)

The scheduler attempts to mitigate the second item by cancelling the backoff period if an offer arrives that fits a pod-in-waiting.
However, there is nothing that the scheduler can do if there are no resources available in the cluster.

That said, the current scheduling algorithm is naive: it makes **no attempts to pack multiple pods into a single offer**.
This means that each pod launch requires an independent offer.
In a small cluster resource offers do not arrive very frequently.
In a large cluster with a "decent" amount of free resources the arrival rate of offers is expected to be much higher.

The slave on each host announces offers to Mesos periodically.
In a single node cluster only a single slave process is advertising resources to the master.
The master will pass those along to the scheduler, at some interval and level of 'fairness' determined by mesos.
That scheduler will pair each resource offer with a pod that needs to be placed in the cluster.
Once paired, a task is launched to instantiate the pod.
The used resources will be marked as consumed, the remaining resources are "returned" to the cluster and the scheduler will wait for the next resource offer from the master... and the cycle repeats itself.
This likely limits the scheduling throughput observable in a single-node cluster.

The team plans to conduct benchmarks on the scheduling algorithm to establish some baselines, and is definitely thinking about ways to increase scheduling throughput- including scheduling multiple pods per offer.

#### Runtime Configuration

- mesos: `--offer_timeout` : Duration of time before an offer is rescinded from a framework.
  This helps fairness when running frameworks that hold on to offers, or frameworks that accidentally drop offers.
  ([via](http://mesos.apache.org/documentation/latest/configuration/))
- k8s-mesos `--scheduler-config` : An ini-style configuration file with low-level scheduler settings.
  See `offer-ttl`, `initial-pod-backoff`, and `max-pod-backoff`.
  ([via](https://github.com/kubernetes/kubernetes/blob/master/contrib/mesos/pkg/scheduler/config/config.go))

What is not configurable, but perhaps should be, are the mesos "filters" that the scheduler includes when declining offers that are not matched to pods within the configured `offer-ttl` (see https://github.com/apache/mesos/blob/0.25.0/include/mesos/mesos.proto#L1165): the current `refuse_seconds` value is hard-coded to 5s.
That parameter should probably be exposed via the scheduler fine tuning mechanism.

#### Backoff

If no matching resource offer can be found for a pod then that pod is put into a backoff queue.
Once the backoff period expires the pod is re-added to the scheduling queue.
The backoff period may be truncated by the arrival of an offer with matching resources.
This is an event-based design and there is no polling.

#### Debugging

Good insight may be achieved when all of the relevant logs are collected into a single tool (Splunk, or an ELK stack) in a manner such that it is trivial to search for something along the lines of a task-id or pod-id during cluster debugging sessions.

The scheduler also offers `/debug` API endpoints that may be useful:
- on-demand explicit reconciliation: /debug/actions/requestExplicit
- on-demand implicit reconciliation: /debug/actions/requestImplicit
- kamikaze (terminate all "empty" executors that aren't running pods): /debug/actions/kamikaze
- pods to be scheduled: /debug/scheduler/podqueue
- pod registry changes waiting to be processed: /debug/scheduler/podstore
- schedulers internal task registry state: /debug/registry/tasks
- scheduler metrics are available at /metrics

## DCOS Package Known Issues

All of the issues in the above section also apply to the Kubernetes-Mesos DCOS package builds.
The issues listed in this section apply specifically to the Kubernetes-Mesos DCOS package available from https://github.com/mesosphere/multiverse.

### Etcd

The default configuration of the DCOS Kubernetes package launches an internal etcd process **which only persists the cluster state in the sandbox of the current container instance**. While this is simpler for the first steps with Kubernetes-Mesos, it means that any cluster state is lost when the Kubernetes-Mesos Docker container is restarted.

Hence, for any kind of production-like deployment it is highly recommended to install the etcd DCOS package alongside Kubernetes-Mesos and
configure the later to use the etcd cluster. Further instructions
can be found at https://docs.mesosphere.com/services/kubernetes/#install.

This situation will eventually go away as soon as DCOS supports package dependencies and/or interactive package configuration.

### Kubectl

The following `kubectl` and `dcos kubectl` commands are not yet supported:

- exec (see [#356](https://github.com/mesosphere/kubernetes-mesos/issues/356))
- logs (see [#587](https://github.com/mesosphere/kubernetes-mesos/issues/587))
- port-forward
- proxy


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/issues.md?pixel)]()
