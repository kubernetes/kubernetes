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

# Network Plugins

__Disclaimer__: Network plugins are in alpha. Its contents will change rapidly.

Network plugins in Kubernetes come in a few flavors:
* Plain vanilla exec plugins - deprecated in favor of CNI plugins.
* CNI plugins: adhere to the appc/CNI specification, designed for interoperability.
* Kubenet plugin: implements basic `cbr0` using the `bridge` and `host-local` CNI plugins

## Installation

The kubelet has a single default network plugin, and a default network common to the entire cluster. It probes for plugins when it starts up, remembers what it found, and executes the selected plugin at appropriate times in the pod lifecycle (this is only true for docker, as rkt manages its own CNI plugins). There are two Kubelet command line parameters to keep in mind when using plugins:
* `network-plugin-dir`: Kubelet probes this directory for plugins on startup
* `network-plugin`: The network plugin to use from `network-plugin-dir`.  It must match the name reported by a plugin probed from the plugin directory.  For CNI plugins, this is simply "cni".

## Network Plugin Requirements

Besides providing the [`NetworkPlugin` interface](../../pkg/kubelet/network/plugins.go) to configure and clean up pod networking, the plugin may also need specific support for kube-proxy.  The iptables proxy obviously depends on iptables, and the plugin may need to ensure that container traffic is made available to iptables.  For example, if the plugin connects containers to a Linux bridge, the plugin must set the `net/bridge/bridge-nf-call-iptables` sysctl to `1` to ensure that the iptables proxy functions correctly.  If the plugin does not use a Linux bridge (but instead something like Open vSwitch or some other mechanism) it should ensure container traffic is appropriately routed for the proxy.

By default if no kubelet network plugin is specified, the `noop` plugin is used, which sets `net/bridge/bridge-nf-call-iptables=1` to ensure simple configurations (like docker with a bridge) work correctly with the iptables proxy.

### Exec

Place plugins in `network-plugin-dir/plugin-name/plugin-name`, i.e if you have a bridge plugin and `network-plugin-dir` is `/usr/lib/kubernetes`, you'd place the bridge plugin executable at `/usr/lib/kubernetes/bridge/bridge`. See [this comment](../../pkg/kubelet/network/exec/exec.go) for more details.

### CNI

The CNI plugin is selected by passing Kubelet the `--network-plugin=cni` command-line option.  Kubelet reads the first CNI configuration file from `--network-plugin-dir` and uses the CNI configuration from that file to set up each pod's network.  The CNI configuration file must match the [CNI specification](https://github.com/appc/cni/blob/master/SPEC.md), and any required CNI plugins referenced by the configuration must be present in `/opt/cni/bin`.

### kubenet

The Linux-only kubenet plugin provides functionality similar to the `--configure-cbr0` kubelet command-line option.  It creates a Linux bridge named `cbr0` and creates a veth pair for each pod with the host end of each pair connected to `cbr0`.  The pod end of the pair is assigned an IP address allocated from a range assigned to the node through either configuration or by the controller-manager.  `cbr0` is assigned an MTU matching the smallest MTU of an enabled normal interface on the host.  The kubenet plugin is currently mutually exclusive with, and will eventually replace, the --configure-cbr0 option.  It is also currently incompatible with the flannel experimental overlay.

The plugin requires a few things:
* The standard CNI `bridge` and `host-local` plugins to be placed in `/opt/cni/bin`.
* Kubelet must be run with the `--network-plugin=kubenet` argument to enable the plugin
* Kubelet must also be run with the `--reconcile-cidr` argument to ensure the IP subnet assigned to the node by configuration or the controller-manager is propagated to the plugin
* The node must be assigned an IP subnet through either the `--pod-cidr` kubelet command-line option or the `--allocate-node-cidrs=true --cluster-cidr=<cidr>` controller-manager command-line options.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/network-plugins.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
