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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/admin/network-plugins.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Network Plugins

__Disclaimer__: Network plugins are in alpha. Its contents will change rapidly.

Network plugins in Kubernets come in 2 flavors:
* Plain vanilla exec plugins - deprecated in favor of CNI plugins.
* CNI plugins - adhere to the appc/CNI specification, designed for interoperability.

## Installation

The kubelet has a single default network plugin, and a default network common to the entire cluster. It probes for plugins when it starts up, remembers what it found, and executes the plugin at appropriate times in the pod lifecycle (this is true for docker, rkt manages its own CNI plugins). There are 2 Kubelet command line parameters to keep in mind when installing plugins:
* network-plugin-dir: The directory the Kubelet probes for plugins. For exec plugins this directory must contain the plugin binaries as described below. For CNI plugins this directory can contain a networking .conf file.
* network-plugin-name: Must match the name reported by the plugin object created by the Kubelet when it probes for plugins. For CNI plugins, this is simply "cni".

### Exec

Place plugins in network-plugin-dir/plugin-name/plugin-name, i.e if you have a bridge plugin and network-plugin-dir is /usr/lib/kubernetes, you'd place the bridge plugin in /usr/lib/kubernetes/bridge/bridge. See [this comment](../../pkg/kubelet/network/exec/exec.go) for more details.

### CNI

CNI plugins require 1 or more netwwork configuration files, and some executables that can fulfill the network configuration. The network configuration must match the CNI specification. To install a CNI plugin place the .conf file in network-plugin-dir and the binaries that fulfill the plugin in `/opt/cni/bin`.

## Using the default Kubernetes network plugin

To use the default Kubernetes network plugin:
1. Specify a `--network-plugin-dir`: You *do not* need to place anything in this directory. It's the directory the Kubelet will write out any network.conf files it needs to, to fulfill the plugin it chooses.
2. Specify `--network-plugin-name=net.alpha.kubernetes.io/default`, where kubernetes.io/default matches the [DefaultPluginName](). This instructs the Kubelet to write a net.conf, and execute a (CNI) plugin in `/opt/cni/bin` (configurable with a vendor directory).
3. Set `--configure-cbr0`: This gives the Kubelet control of the container bridge. Depending on the network plugin that satisfies the default, it may or may not need this.

To clarify, the difference between this mode and the "normal" CNI mode, is that the latter has the same specifications but --network-plugin-dir actually contains a net.conf, and --network-plugin-name cannot be `net.alpha.kubernetes.io/default`. If you specify `net.alpha.kubernets.io/default` AND put your own net.conf in `--network-plugin-dir`, you will end up with some undefined behaviour (currently, the kubelet will ignore your conf and write out its own).

## Writing a network plugin

This section is very much a work in progress.

Each of the above plugin specifications has a Kubelet-shim, a small interface that sits between the Kubelet and the actual plugin. This shim must implment the methods of a [NetworkPlugin](../../pkg/kubelet/network/plugins.go).

### How is the default Kubelet network plugin chosen?

The Kubelet currently does the following dance:

```
if --network-name starts with net.alpha.kubernetes.io
  kubelet.networkname = --network-name
  --network-name=cni
probe plugins using --network-name=cni
  no net conf found, but that's ok
  load shell CNI network plugin
if node.podCIDR
  write bridge netconf to --network-plugin-dir
start creating pods
```

### At what points in the pod lifecycle are the plugins invoked?

For each non-host networking pod the Kubelet does the following:

```
Create pause container
   Invoke bridge plugin from /opt/cni/bin with netconf written by kubelet
     Bridge plugin invokes host-local IPAM to allocate podCIDR
Create other containers
...

Periodically
  nsenter pause container and return ip
  sync with apiserver podip on change
...
Delete pause container
   Invoke bridge plugin from /opt/cni/bin with netconf written by kubelet
     Bridge plugin invokes host-local IPAM to de-allocate podCIDR
Delete other containers
```

### How to get more inforamtion from the runtime?

After choosing a default plugin, the Kubelet invokes the network interface's `Init` method with a Kubernetes host interface capable of retrieving more information from both the container runtime and/or apiserver.

## Limitations

* Docker's and your network plugin's bridge name must be different. Docker will startup by default on docker0, the plugin's bridge name is in the netconf written by kubelet (hardcoded to "cbr0" for now).
* Only the bridge plugin is currently supported, both `net.alpha.kubernetes.io/default` and `net.alpha.kubernetes.io/bridge` will end up using bridge.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/network-plugins.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
