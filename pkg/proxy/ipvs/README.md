# How to use IPVS

This document shows how to use kube-proxy ipvs mode.

### What is IPVS

**IPVS (IP Virtual Server)** implements transport-layer load balancing, usually called Layer 4 LAN switching, as part of
Linux kernel.

IPVS runs on a host and acts as a load balancer in front of a cluster of real servers. IPVS can direct requests for TCP
and UDP-based services to the real servers, and make services of real servers appear as irtual services on a single IP address.

### How to use

##### Load IPVS kernel modules

Currently the IPVS kernel module can't be loaded automatically, so first we should use the following command to load IPVS kernel
modules manually.

```shell
modprobe ip_vs
modprobe ip_vs_rr
modprobe ip_vs_wrr
modprobe ip_vs_sh
modprobe nf_conntrack_ipv4
```

After that, use `lsmod | grep ip_vs` to make sure kernel modules are loaded.

##### Run kube-proxy in ipvs mode

First, [run cluster locally](https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md). 

By default kube-proxy will run in iptables mode, with configuration file `/tmp/kube-proxy.yaml`. so we need to change the
configuration file and restart it. Here is a yaml file for reference.

```yaml
apiVersion: componentconfig/v1alpha1
kind: KubeProxyConfiguration
clientConnection:
  kubeconfig: /var/run/kubernetes/kube-proxy.kubeconfig
hostnameOverride: 127.0.0.1
mode: ipvs
featureGates: AllAlpha=true
ipvs:
  minSyncPeriod: 10s
  syncPeriod: 60s
```

##### Test

Use `ipvsadm` tool to test whether the kube-proxy start succeed. By default we may get result like:

```shell
# ipvsadm -ln
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
TCP  10.0.0.1:443 rr persistent 10800
  -> 10.229.43.2:6443             Masq    1      0          0         
TCP  10.0.0.10:53 rr      
UDP  10.0.0.10:53 rr
```

