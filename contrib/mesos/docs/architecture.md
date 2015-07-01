# Kubernetes-Mesos Architecture

An [Apache Mesos][1] cluster consists of one or more masters, and one or more slaves.
Kubernetes-Mesos (k8sm) operates as a Mesos framework that runs on the cluster.
As a framework, k8sm provides scheduler and executor components, both of which are hybrids of Kubernetes and Mesos:
the scheduler component integrates the Kubernetes scheduling API and the Mesos scheduler runtime, whereas;
the executor component integrates Kubernetes kubelet services and the Mesos executor runtime.

Multiple Mesos masters are typically configured to coordinate leadership election via Zookeeper.
Future releases of Mesos may implement leader election protocols [differently][2].
Kubernetes maintains its internal registry (pods, replication controllers, bindings, minions, services) in etcd.
Users typically interact with Kubernetes using the `kubectl` command to manage Kubernetes primitives.

When a pod is created in Kubernetes, the k8sm scheduler creates an associated Mesos task and queues it for scheduling.
Upon pairing the pod/task with an acceptable resource offer, the scheduler binds the pod/task to the offer's slave.
As a result of binding the pod/task is launched and delivered to an executor (an executor is created by the Mesos slave if one is not already running).
The executor launches the pod/task, which registers the bound pod with the kubelet engine and the kubelet begins to manage the lifecycle of the pod instance.

![Architecture Diagram](architecture.png)

## Networking

Kubernetes-Mesos uses "normal" Docker IPv4, host-private networking, rather than Kubernetes' SDN-based networking that assigns an IP per pod. This is mostly transparent to the user, especially when using the service abstraction to access pods. For details on some issues it creates, see [issues][3].

![Network Diagram](networking.png)

[1]: http://mesos.apache.org/
[2]: https://issues.apache.org/jira/browse/MESOS-1806
[3]: issues.md#service-endpoints

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/README.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/architecture.md?pixel)]()
