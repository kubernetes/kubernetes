### Objective
The goal of this proposal is to define a simplified flex volume plugin driver installation model.
Existing plugin driver installation model has the following drawbacks.
* It requires manual installation on every node and restart of kubelet on every node & controller-manager on the master node.
* It is not supported in GCE & CoreOS environments where access to master node or root filesystem is restricted. Access to master node is required to install the plugin for controller-manager to provision, delete, attach & detach calls.
* Manual and controlled upgrade on every node and requires downtime.

The new model fixes these shortcomings. Installation of drivers is no longer manual and does not require restart of any Kubernetes modules. Upgrade is as simple as upgrading a daemon-set in many cases.

Please refer to proposal (#33538) for FlexVolumeDriver API type.

## New proposal:
### Discovery:
Kubelet & Controller manager discovers drivers looking for them in /run/kubernetes/plugins/volume/grpc directory. The discovery is dynamic and drivers are discovered only when a pod uses them.

Drivers are expected to create a .sock file(UNIX domain sockets) in /run/kubernetes/plugins/volume/grpc directory. The name of the .sock file determines the name of the driver.

The following vendor driver naming convention is followed to create a .sock file.

```
/run/kubernetes/plugins/volume/grpc/<vendor~driver>/driver.sock
```

where **vendor/driver** is the actual driver name.

Example:
	driverName = nfs
	vendorName = ganesha
   then the driver is expected to create .sock in ```/run/kubernetes/plugins/volume/grpc/ganesha-nfs/nfs.sock```

### Lifecycle:
Drivers run on every Kubernetes node (except master nodes) and create corresponding .sock files. It is recommended to use the Kubernetes daemon-set construct to start drivers on every node in the cluster.

Also, it is recommended to activate the plugins using systemd or some other socket activation models.

### Upgrade
In most cases upgrade is as simple as upgrading a daemon-set. But in some cases (e.g. fuse based drivers), driver pod cannot be restarted when it is in use by a Pod. In those cases, it is recommended to separate out the pod running the server/fuse backend into a separate dameon-set

###Example workflow:
#### Installation
* Build a container image with support for one or more plugin drivers.
* Run the container as a daemon-set. The pod created by daemon-set is expected to create .sock files for all the plugins it supports in /run/kubernetes/plugins/volume/grpc directory.

#### Usage
* Create a pod with persistent volume using one of the drivers.

### Open items
* This approach supports REMOTE attachment policy where the volumes are attached to a remote node by controller-manager only if the daemon-set pods are able to run on master nodes. In environments like GCE, no daemon-sets/pods can be run on master nodes. So, this approch will not work on GCE. I am still working on closing this gap.

