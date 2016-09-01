<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

[Sysdig Cloud](http://www.sysdig.com/) is a monitoring, alerting, and troubleshooting platform designed to natively support containerized and service-oriented applications.

Sysdig Cloud comes with built-in, first class support for Kubernetes. In order to instrument your Kubernetes environment with Sysdig Cloud, you simply need to install the Sysdig Cloud agent container on each underlying host in your Kubernetes cluster. Sysdig Cloud will automatically begin monitoring all of your hosts, apps, pods, and services, and will also automatically connect to the Kubernetes API to pull relevant metadata about your environment.

# Example Installation Files

Provided here are two example sysdig.yaml files that can be used to automatically deploy the Sysdig Cloud agent container across a Kubernetes cluster.

The recommended method is using daemon sets - minimum kubernetes version 1.1.1.

If daemon sets are not available, then the replication controller method can be used (based on [this hack](https://stackoverflow.com/questions/33377054/how-to-require-one-pod-per-minion-kublet-when-configuring-a-replication-controll/33381862#33381862 )).

# Latest Files

See here for the latest maintained and updated versions of these example files:
https://github.com/draios/sysdig-cloud-scripts/tree/master/agent_deploy/kubernetes

# Install instructions

Please see the Sysdig Cloud support site for the latest documentation:
http://support.sysdigcloud.com/hc/en-us/sections/200959909





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/sysdig-cloud/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
