# Calico Policy Controller
==============

Calico is an implementation of the Kubernetes network policy API.  The provided manifest installs a DaemonSet which runs Calico on each node in the cluster.

### Templating

The provided `calico-node.yaml` manifest includes the following placeholders which are populated
via templating.

- `__CLUSTER_CIDR__`: The IP range from which Pod IP addresses are assigned.

### Learn More

Learn more about Calico at http://docs.projectcalico.org

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/calico-policy-controller/README.md?pixel)]()
