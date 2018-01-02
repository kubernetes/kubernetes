# Design of runtime reconfiguration

Runtime reconfiguration is one of the hardest and most error prone features in a distributed system, especially in a consensus based system like etcd.

Read on to learn about the design of etcd's runtime reconfiguration commands and how we tackled these problems.

## Two phase config changes keep the cluster safe

In etcd, every runtime reconfiguration has to go through [two phases][add-member] for safety reasons. For example, to add a member, first inform cluster of new configuration and then start the new member.

Phase 1 - Inform cluster of new configuration

To add a member into etcd cluster, make an API call to request a new member to be added to the cluster. This is only way to add a new member into an existing cluster. The API call returns when the cluster agrees on the configuration change.

Phase 2 - Start new member

To join the etcd member into the existing cluster, specify the correct `initial-cluster` and set `initial-cluster-state` to `existing`. When the member starts, it will contact the existing cluster first and verify the current cluster configuration matches the expected one specified in `initial-cluster`. When the new member successfully starts, the cluster has reached the expected configuration.

By splitting the process into two discrete phases users are forced to be explicit regarding cluster membership changes. This actually gives users more flexibility and makes things easier to reason about. For example, if there is an attempt to add a new member with the same ID as an existing member in an etcd cluster, the action will fail immediately during phase one without impacting the running cluster. Similar protection is provided to prevent adding new members by mistake. If a new etcd member attempts to join the cluster before the cluster has accepted the configuration change,, it will not be accepted by the cluster.

Without the explicit workflow around cluster membership etcd would be vulnerable to unexpected cluster membership changes. For example, if etcd is running under an init system such as systemd, etcd would be restarted after being removed via the membership API, and attempt to rejoin the cluster on startup. This cycle would continue every time a member is removed via the API and systemd is set to restart etcd after failing, which is unexpected.

We expect runtime reconfiguration to be an infrequent operation. We decided to keep it explicit and user-driven to ensure configuration safety and keep the cluster always running smoothly under explicit control.

## Permanent loss of quorum requires new cluster

If a cluster permanently loses a majority of its members, a new cluster will need to be started from an old data directory to recover the previous state.

It is entirely possible to force removing the failed members from the existing cluster to recover. However, we decided not to support this method since it bypasses the normal consensus committing phase, which is unsafe. If the member to remove is not actually dead or force removed through different members in the same cluster, etcd will end up with a diverged cluster with same clusterID. This is very dangerous and hard to debug/fix afterwards. 

With a correct deployment, the possibility of permanent majority lose is very low. But it is a severe enough problem that worth special care. We strongly suggest reading the [disaster recovery documentation][disaster-recovery] and prepare for permanent majority lose before putting etcd into production.

## Do not use public discovery service for runtime reconfiguration

The public discovery service should only be used for bootstrapping a cluster. To join member into an existing cluster, use runtime reconfiguration API. 

Discovery service is designed for bootstrapping an etcd cluster in the cloud environment, when the IP addresses of all the members are not known beforehand. After successfully bootstrapping a cluster, the IP addresses of all the members are known. Technically, the discovery service should no longer be needed.

It seems that using public discovery service is a convenient way to do runtime reconfiguration, after all discovery service already has all the cluster configuration information. However relying on public discovery service brings troubles: 

1. it introduces external dependencies for the entire life-cycle of the cluster, not just bootstrap time. If there is a network issue between the cluster and public discovery service, the cluster will suffer from it.
 
2. public discovery service must reflect correct runtime configuration of the cluster during it life-cycle. It has to provide security mechanism to avoid bad actions, and it is hard. 

3. public discovery service has to keep tens of thousands of cluster configurations. Our public discovery service backend is not ready for that workload.

To have a discovery service that supports runtime reconfiguration, the best choice is to build a private one.

[add-member]: runtime-configuration.md#add-a-new-member
[disaster-recovery]: recovery.md
