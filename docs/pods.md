# Pods

A _pod_ (as in a pod of whales or pea pod) is a relatively tightly coupled group of containers that are scheduled onto the same host. It models an application-specific "virtual host" in a containerized environment. Pods serve as units of scheduling, deployment, and horizontal scaling/replication, and share fate.

In addition to defining the containers that run in the pod, the pod specifies a set of shared storage volumes. Pods facilitate data sharing and IPC among their constituents. In the future, they may share CPU and/or memory ([LPC2013](http://www.linuxplumbersconf.org/2013/ocw//system/presentations/1239/original/lmctfy%20(1).pdf)).

The containers in the pod also all use the same network namespace/IP (and port space). The goal is for each pod have an IP address in a flat shared networking namespace that has full communication with other physical computers and containers across the network. [More details on networking](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/networking.md).

While pods can be used to host vertically integrated application stacks, their primary motivation is to support co-located, co-managed helper programs, such as:
- content management systems, file and data loaders, local cache managers, etc.
- log and checkpoint backup, compression, rotation, snapshotting, etc.
- data change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
- proxies, bridges, and adapters
- controllers, managers, configurators, and updaters

Individual pods are not intended to run multiple instances of the same application, in general.

