# Cluster add-ons

Cluster add-ons are Services and Replication Controllers (with pods) that are
shipped with the Kubernetes binaries and are considered an inherent part of the
Kubernetes clusters. The add-ons are visible through the API (they can be listed
using ```kubectl```), but manipulation of these objects is discouraged because
the system will bring them back to the original state, in particular:
* if an add-on is stopped, it will be restarted automatically
* if an add-on is rolling-updated (for Replication Controllers), the system will stop the new version and
  start the old one again (or perform rolling update to the old version, in the
  future).

On the cluster, the add-ons are kept in ```/etc/kubernetes/addons``` on the master node, in yaml files
(json is not supported at the moment). A system daemon periodically checks if
the contents of this directory is consistent with the add-on objects on the API
server. If any difference is spotted, the system updates the API objects
accordingly. (Limitation: for now, the system compares only the names of objects
in the directory and on the API server. So changes in parameters may not be
noticed). So the only persistent way to make changes in add-ons is to update the
manifests on the master server. But still, users are discouraged to do it
on their own - they should rather wait for a new release of
Kubernetes that will also contain new versions of add-ons.

Each add-on must specify the following label: ```kubernetes.io/cluster-service: true```.
Yaml files that do not define this label will be ignored.

The naming convention for Replication Controllers is
```<basename>-<version>```, where ```<basename>``` is the same in consecutive
versions and ```<version>``` changes when the component is updated
(```<version>``` must not contain ```-```). For instance,
```heapster-controller-v1``` and ```heapster-controller-12``` are the
same controllers with two different versions, while ```heapster-controller-v1```
and ```heapster-newcontroller-12``` are treated as two different applications.
When a new version of a Replication Controller add-on is found, the system will
stop the old (current) replication controller and start the new one
(in the future, rolling update will be performed).

For services, the naming scheme is just ```<basename>``` (with empty version number)
because we do not expect the service name to change in consecutive versions (and
rolling-update of services does not exist).

# Add-on update procedure

To update add-ons, just update the contents of ```/etc/kubernetes/addons```
directory with the desired definition of add-ons. Then the system will take care
of:

1. Removing the objects from the API server whose manifest was removed.
  1. This is done for add-ons in the system that do not have a manifest file with the
     same basename
1. Creating objects from new manifests
  1. This is done for manifests that do not correspond to existing API objects
     with the same basename
1. Updating objects whose basename is the same, but whose versions changed.
  1. The update is currently performed by removing the old object and creating
     the new one. In the future, rolling update of replication controllers will
     be implemented to keep the add-on services up and running during update of add-on
     pods.
  1. Note that this cannot happen for Services as their version is always empty.

Note that in order to run the updator script, python is required on the machine.
For OS distros that don't have python installed, a python container will be used.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/README.md?pixel)]()
