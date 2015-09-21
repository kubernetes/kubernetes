## Known Issues

### Pod Placement

The initial plan was to implement pod placement (aka scheduling "constraints") using rules similar to those found in Marathon.
Upon further consideration it has been decided that a greater alignment between the stock Kubernetes scheduler and kubernetes-mesos scheduler would benefit both projects, as well as end-users.
Currently it is not possible to specify pod placement constraints for the kubernetes-mesos scheduler.
This issue is being tracked here: https://github.com/mesosphere/kubernetes-mesos/issues/338

### Ports

Mesos typically defines `ports` resources for each slave and these ports are consumed by tasks, as they are launched, that require one or more host ports.
Kubernetes pod container specifications identify two types of ports, container ports and host ports: 
container ports are allocated from the network namespace of the pod, which is independent from that of the host, whereas;
host ports are allocated from the network namespace of the host.
The k8sm scheduler recognizes the declared host ports of each container in a pod/task and for each such port, attempts to allocate it from the offered ports listed in mesos resource offers.
If no host port is declared, then the scheduler may choose any port from the offered ports ranges.

If slaves are configured to offer a `ports` resource range, for example [31000-32000], then any host ports declared in the pod container specification must fall within that range.
Ports declared outside that range (other than zero) will never match resource offers received by the k8sm scheduler, and so pod specifications that declare such ports will never be executed as tasks on the cluster.

As opposed to Kubernetes proper, a missing pod container host port specification or a host port set to zero will allocate a host port from a resource offer.

### Static Pods

Static pods in general are supported by the k8sm-scheduler. The path of the pod definitions can be set via the `--static-pods-config` flag. There are two restrictions currently in that implementation:

- static pods *must have resource limits* on cpu and memory in their container specs (compare the [k8sm architecture](architecture.md))
- static pods *are read only once* by the k8sm-scheduler on startup. Only newly started executor will get the latest static pod specs from the defined static pod directory.

### Service Endpoints

At the time of this writing both Kubernetes and Mesos are using IPv4 addressing, albeit under different assumptions.
Mesos clusters configured with Docker typically use default Docker networking, which is host-private.
Kubernetes clusters assume a custom Docker networking configuration that assigns a cluster-routable IPv4 address to each pod, meaning that a process running anywhere on a Kubernetes cluster can reach a pod running on the same cluster by using the pod's Docker-assigned IPv4 address.

Kubernetes service endpoints terminate, by default, at a backing pod's IPv4 address using the container-port selected for in the service specification (PodIP:ContainerPort).
This is problematic when default Docker networking has been configured, such as in the case of typical Mesos clusters, because a pod's host-private IPv4 address is not intended to be reachable outside of its host.

The k8sm project has implemented a work-around: service endpoints are terminated at HostIP:HostPort, where the HostIP is the IP address of the Mesos slave and the HostPort is the host port declared in the pod container port specification.
Host ports that are not defined, or else defined as zero, will automatically be assigned a (host) port resource from a resource offer.

To disable the work-around and revert to vanilla Kubernetes service endpoint termination:

* execute the k8sm controller-manager with `-host_port_endpoints=false`

Then the usual Kubernetes network assumptions must be fulfilled for Kubernetes to work with Mesos, i.e. each container must get a cluster-wide routable IP (compare [Kubernetes Networking documentation](../../../docs/design/networking.md#container-to-container)).

Future support for IPv6 addressing in Docker and Kubernetes should obviate the need for this work-around.

### Orphan Pods

The default `executor_shutdown_grace_period` of a Mesos slave is 3 seconds.
When the executor is shut down it forcefully terminates the Docker containers that it manages.
However, if terminating the Docker containers takes longer than the `executor_shutdown_grace_period` then some containers may not get a termination signal at all.
A consequence of this is that some pod containers, previously managed by the framework's executor, will remain running on the slave indefinitely.

There are two work-arounds to this problem:
* Restart the framework and it should terminate the orphaned tasks.
* Adjust the value of `executor_shutdown_grace_period` to something greater than 3 seconds.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/issues.md?pixel)]()
