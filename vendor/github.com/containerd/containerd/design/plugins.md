# containerd Plugin Model

With go 1.8 we now have dynamically loaded plugins via go packages.  This seems to be a very easy and clean way to extend containerd.  It does have the drawback of only working on Linux right now but this is where we see the most need for swapping out defaults. 

## core

To be extended the core of containerd has to provide go packages and interfaces that can be extended with third-party packages.  The core should be small but provide value for people building on top of containerd.

The core should be comprised of the following:

* Snapshotters - Provide way to manage the filesystems of containers and images on a host.
* Runtime - Provide a way to launch containers via the OCI runtime specification.
* Distribution - Provide a way to fetch and push content to external sources/registries.
* Content Store - Provide a generic content addressed store for bridging the gap between registries and snapshotters.
* Metadata - Provide a consistent way for the core and various subsystems to store metadata.
* Monitoring - Provide a way to monitor different subsystems, containers, and operations throughout the core with metrics and events.

### Runtime

The runtime code in the core provides API to create, list, and manage containers on the system.  It provides a runtime type that is responsible for creating, deleting, and loading containers.

### Monitoring 

The monitoring subsystem is a way to collect events and metrics from various subsystems.
With the monitoring subsystem you can monitor various types, subsystems, and objects within the core.
This can be used to collect metrics for containers and monitor OOM events when supported.
An example of this is a prometheus monitor that exports container metrics such as cpu, memory, io, and network information.

```go
type ContainerMonitor interface {
	Monitor(context.Context, Container) error 
}
```
