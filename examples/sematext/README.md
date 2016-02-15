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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Sematext Docker Agent Example

This example shows how to run [Sematext Docker Agent](https://github.com/sematext/sematext-agent-docker) as a DaemonSet on an existing Kubernetes cluster.

This example will create a DaemonSet which places Sematext Docker Agent on every node in the cluster. 

It will collect Host Metrics of each Node, Container Metrics and Logs from all containers, all Docker Events.

### Step 0: Prerequisites

If you are using a Salt based KUBERNETES\_PROVIDER (**gce**, **vagrant**, **aws**), you should make sure the creation of privileged containers via the API is enabled. Check `cluster/saltbase/pillar/privilege.sls`.

DaemonSets must be enabled on your cluster. Instructions for enabling DaemonSet can be found [here](../../docs/api.md#enabling-the-extensions-group).

### Step 1: Configure Sematext Docker Agent

Sematext Docker Agent is configured via environment variables.

The [Sematext Docker Agent Github page]
(https://github.com/sematext/sematext-agent-docker) lists all the other options. E.g. to filter metrics and logs only for specific pods/images/containers.

1. Get a free account [apps.sematext.com](https://apps.sematext.com/users-web/register.do), if you don't have one already ...  
2. [Create an SPM App of type “Docker”](https://apps.sematext.com/spm-reports/registerApplication.do) to obtain the SPM Application Token
3. Create a [Logsene](http://www.sematext.com/logsene/) App to obtain the Logsene Token
4. Edit the values of LOGSENE_TOKEN and SPM_TOKEN in the DaemonSet definition.

Please note the env variable KUBERNETES=1 to add Pod name and namespace to container logs.

<!-- BEGIN MUNGE: EXAMPLE sematext-agent-daemonset.yml -->


<!-- END MUNGE: EXAMPLE sematext-agent-daemonset.yml -->
[Download example](sematext-agent-daemonset.yml?raw=true)


### Step 2: Run the DaemonSet.

The DaemonSet definition instructs Kubernetes to place a Sematet Docker Agent on each Kubernetes node. Create the DaemonSet to activate Sematext Agent Docker:

```
kubectl create -f sematext-agent-daemonset.yml
```

One minute after the deployemnet you should see your metrics in [the user interface](https://apps.sematext.com). 

