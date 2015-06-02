# Cluster add-ons

Cluster add-ons are Services and Replication Controllers (with pods) that are
shipped with the kubernetes binaries and whose update policy is also consistent
with the update of kubernetes cluster.

On the clusterm the addons are kept in ```/etc/kubernetes/addons``` on the master node, in yaml files
(json is not supported at the moment).
Each add-on must specify the following label: ````kubernetes.io/cluster-service: true````.
Yaml files that do not define this label will be ignored.

The naming convention for Replication Controllers is
```<basename>-<version>```, where ```<basename>``` is the same in consecutive
versions and ```<version>``` changes when the component is updated
(```<version>``` must not contain ```-```). For instance,
```heapseter-controller-v1``` and ```heapster-controller-12``` are the
same controllers with two different versions, while ```heapseter-controller-v1```
and ```heapster-newcontroller-12``` are treated as two different applications.
For services it is just ```<basename>``` (with empty version number)
because we do not expect the service
name to change in consecutive versions. The naming convetion is important for add-on update.

# Add-on update

To update add-ons, just update the contents of ```/etc/kubernetes/addons```
directory with the desired definition of add-ons. Then the system will take care
of:

1. Removing the objects from the API server whose manifest was removed.
  1. This is done for add-ons in the system that do not have a manifest file with the
     same basename
1. Creating objects from new manifests
  1. This is done for manifests that do not correspond to existing API objects
     with the same basename
1. Updating objects whose basename is the samem but whose versions changed.
  1. The update is currently performed by removing the old object and creating
     the new one. In the future, rolling update of replication controllers will
     be implemented to keep the add-on services up and running during update of add-on
     pods.
  1. Note that this cannot happen for Services as their version is always empty.




[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/README.md?pixel)]()
