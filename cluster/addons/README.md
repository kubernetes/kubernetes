# Cluster add-ons

Cluster add-ons are resources like Services and Deployments (with pods) that are
shipped with the Kubernetes binaries and are considered an inherent part of the
Kubernetes clusters. The add-ons are visible through the API (they can be listed using
`kubectl`), but direct manipulation of these objects through Apiserver is discouraged
because the system will bring them back to the original state, in particular:
- If an add-on is deleted, it will be recreated automatically.
- If an add-on is updated through Apiserver, it will be reconfigured to the state given by
the supplied fields in the initial config.

On the cluster, the add-ons are kept in `/etc/kubernetes/addons` on the master node, in
yaml / json files. The addon manager periodically `kubectl apply`s the contents of this
directory. Any legit modification would be reflected on the API objects accordingly.
Particularly, rolling-update for deployments is now supported.

Each add-on must specify the following label: `kubernetes.io/cluster-service: true`.
Config files that do not define this label will be ignored. For those resources
exist in `kube-system` namespace but not in `/etc/kubernetes/addons`, addon manager
will attempt to remove them if they are attached with this label. Currently the other
usage of `kubernetes.io/cluster-service` is for `kubectl cluster-info` command to recognize
these cluster services.

The suggested naming for most types of resources is just `<basename>` (with no version
number) because we do not expect the resource name to change. But resources like `Pod`
, `ReplicationController` and `DaemonSet` are exceptional. As `Pod` updates may not change
fields other than `containers[*].image` or `spec.activeDeadlineSeconds` and may not add or
remove containers, it may not be sufficient during a major update. For `ReplicationController`,
most of the modifications would be legit, but the underlying pods would not got re-created
automatically. `DaemonSet` has similar problem as the `ReplicationController`. In these
cases, the suggested naming is `<basename>-<version>`. When version changes, the system will
delete the old one and create the new one (order not guaranteed).

# Add-on update procedure

To update add-ons, just update the contents of `/etc/kubernetes/addons`
directory with the desired definition of add-ons. Then the system will take care
of:

- Removing objects from the API server whose manifest was removed.
- Creating objects from new manifests
- Updating objects whose fields are legally changed.

# Cooperating with Horizontal / Vertical Auto-Scaling

As all cluster add-ons will be reconciled to the original state given by the initial config.
In order to make Horizontal / Vertical Auto-scaling functional, the related fields in config should
be left unset. More specifically, leave `replicas` in `ReplicationController` / `Deployment`
/ `ReplicaSet` unset for Horizontal Scaling, and leave `resources` for container unset for Vertical
Scaling. The periodical update won't include these specs, which will be managed by Horizontal / Vertical
 Auto-scaler.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/README.md?pixel)]()
