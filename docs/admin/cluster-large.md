<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/admin/cluster-large.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Large Cluster

## Support

At v1.0, Kubernetes supports clusters up to 100 nodes with 30 pods per node and 1-2 containers per pod.

## Setup

A cluster is a set of nodes (physical or virtual machines) running Kubernetes agents, managed by a "master" (the cluster-level control plane).

Normally the number of nodes in a cluster is controlled by the the value `NUM_MINIONS` in the platform-specific `config-default.sh` file (for example, see [GCE's `config-default.sh`](http://releases.k8s.io/HEAD/cluster/gce/config-default.sh)).

Simply changing that value to something very large, however, may cause the setup script to fail for many cloud providers. A GCE deployment, for example, will run in to quota issues and fail to bring the cluster up.

When setting up a large Kubernetes cluster, the following issues must be considered.

### Quota Issues

To avoid running into cloud provider quota issues, when creating a cluster with many nodes, consider:
* Increase the quota for things like CPU, IPs, etc.
  * In [GCE, for example,](https://cloud.google.com/compute/docs/resource-quotas) you'll want to increase the quota for:
    * CPUs
    * VM instances
    * Total persistent disk reserved
    * In-use IP addresses
    * Firewall Rules
    * Forwarding rules
    * Routes
    * Target pools
* Gating the setup script so that it brings up new node VMs in smaller batches with waits in between, because some cloud providers rate limit the creation of VMs.

### Addon Resources

To prevent memory leaks or other resource issues in [cluster addons](../../cluster/addons/) from consuming all the resources available on a node, Kubernetes sets resource limits on addon containers to limit the CPU and Memory resources they can consume (See PR [#10653](http://pr.k8s.io/10653/files) and [#10778](http://pr.k8s.io/10778/files)).

For example:

```yaml
containers:
  - image: gcr.io/google_containers/heapster:v0.15.0
    name: heapster
    resources:
      limits:
        cpu: 100m
        memory: 200Mi
```

These limits, however, are based on data collected from addons running on 4-node clusters (see [#10335](http://issue.k8s.io/10335#issuecomment-117861225)). The addons consume a lot more resources when running on large deployment clusters (see [#5880](http://issue.k8s.io/5880#issuecomment-113984085)). So, if a large cluster is deployed without adjusting these values, the addons may continuously get killed because they keep hitting the limits.

To avoid running into cluster addon resource issues, when creating a cluster with many nodes, consider the following:
* Scale memory and CPU limits for each of the following addons, if used, along with the size of cluster (there is one replica of each handling the entire cluster so memory and CPU usage tends to grow proportionally with size/load on cluster):
  * Heapster ([GCM/GCL backed](http://releases.k8s.io/HEAD/cluster/addons/cluster-monitoring/google/heapster-controller.yaml), [InfluxDB backed](http://releases.k8s.io/HEAD/cluster/addons/cluster-monitoring/influxdb/heapster-controller.yaml), [InfluxDB/GCL backed](http://releases.k8s.io/HEAD/cluster/addons/cluster-monitoring/googleinfluxdb/heapster-controller-combined.yaml), [standalone](http://releases.k8s.io/HEAD/cluster/addons/cluster-monitoring/standalone/heapster-controller.yaml))
  * [InfluxDB and Grafana](http://releases.k8s.io/HEAD/cluster/addons/cluster-monitoring/influxdb/influxdb-grafana-controller.yaml)
  * [skydns, kube2sky, and dns etcd](http://releases.k8s.io/HEAD/cluster/addons/dns/skydns-rc.yaml.in)
  * [Kibana](http://releases.k8s.io/HEAD/cluster/addons/fluentd-elasticsearch/kibana-controller.yaml)
* Scale number of replicas for the following addons, if used, along with the size of cluster (there are multiple replicas of each so increasing replicas should help handle increased load, but, since load per replica also increases slightly, also consider increasing CPU/memory limits):
  * [elasticsearch](http://releases.k8s.io/HEAD/cluster/addons/fluentd-elasticsearch/es-controller.yaml)
* Increase memory and CPU limits slightly for each of the following addons, if used, along with the size of cluster (there is one replica per node but CPU/memory usage increases slightly along with cluster load/size as well):
  * [FluentD with ElasticSearch Plugin](http://releases.k8s.io/HEAD/cluster/saltbase/salt/fluentd-es/fluentd-es.yaml)
  * [FluentD with GCP Plugin](http://releases.k8s.io/HEAD/cluster/saltbase/salt/fluentd-gcp/fluentd-gcp.yaml)

For directions on how to detect if addon containers are hitting resource limits, see the [Troubleshooting section of Compute Resources](../user-guide/compute-resources.md#troubleshooting).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/cluster-large.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
