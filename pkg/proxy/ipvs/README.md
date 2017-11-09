# How to use IPVS

This document shows how to use kube-proxy ipvs mode.

## What is IPVS

**IPVS (IP Virtual Server)** implements transport-layer load balancing, usually called Layer 4 LAN switching, as part of
Linux kernel.

IPVS runs on a host and acts as a load balancer in front of a cluster of real servers. IPVS can direct requests for TCP
and UDP-based services to the real servers, and make services of real servers appear as irtual services on a single IP address.

## How to use

#### Load IPVS kernel modules

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

#### Run kube-proxy in ipvs mode

#### Local UP Cluster

Kube-proxy will run in iptables mode by default in a [local-up cluster](https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md). 

Users should export the env `KUBEPROXY_MODE=ipvs` to specify the ipvs mode before deploying the cluster if want to run kube-proxy in ipvs mode.

#### Cluster Created by Kubeadm

Kube-proxy will run in iptables mode by default in a cluster deployed by [kubeadm](https://kubernetes.io/docs/setup/independent/create-cluster-kubeadm/). 

Since IPVS mode is still feature-gated, users should add the flag `--feature-gates=SupportIPVSProxyMode=true` in `kubeadm init` command

```
kubeadm init --feature-gates=SupportIPVSProxyMode=true
```

to specify the ipvs mode before deploying the cluster if want to run kube-proxy in ipvs mode.

If you are using kubeadm with a configuration file, you can specify the ipvs mode adding `SupportIPVSProxyMode: true` below the `featureGates` field.
Then the configuration file is similar to:

```json
kind: MasterConfiguration
apiVersion: kubeadm.k8s.io/v1alpha1
...
featureGates:
  SupportIPVSProxyMode: true
...
```

#### Test

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
