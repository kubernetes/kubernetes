# How to use IPVS

This document shows how to use kube-proxy ipvs mode.

## What is IPVS

**IPVS (IP Virtual Server)** implements transport-layer load balancing, usually called Layer 4 LAN switching, as part of
Linux kernel.

IPVS runs on a host and acts as a load balancer in front of a cluster of real servers. IPVS can direct requests for TCP
and UDP-based services to the real servers, and make services of real servers appear as virtual services on a single IP address.

## Run kube-proxy in ipvs mode

Currently, local-up scripts and kubeadm support switching IPVS proxy mode via exporting environment variables or specifying flags.

### Local UP Cluster

Kube-proxy will run in iptables mode by default in a [local-up cluster](https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md). 

Users should export the env `KUBE_PROXY_MODE=ipvs` to specify the ipvs mode before deploying the cluster if want to run kube-proxy in ipvs mode.

### Cluster Created by Kubeadm

Kube-proxy will run in iptables mode by default in a cluster deployed by [kubeadm](https://kubernetes.io/docs/setup/independent/create-cluster-kubeadm/). 

If you are using kubeadm with a [configuration file](https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-init/#config-file), you can specify the ipvs mode adding `SupportIPVSProxyMode: true` below the `kubeProxy` field.
Then the configuration file is similar to:

```json
kind: MasterConfiguration
apiVersion: kubeadm.k8s.io/v1alpha1
...
kubeProxy:
  config:
    featureGates: SupportIPVSProxyMode=true
    mode: ipvs
...
```

## Debug

### Check IPVS proxy rules

People can use `ipvsadm` tool to check whether kube-proxy are maintaining IPVS rules correctly. For example, we may get IPVS proxy rules like:

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

### Why kube-proxy can't start IPVS mode

People can do the following check list step by step:

**1. Enable IPVS feature gateway**

Currently IPVS-based kube-proxy is still in alpha phase, people need to enable `--feature-gates=SupportIPVSProxyMode=true` explicitly.

**2. Specify proxy-mode=ipvs**

Tell kube-proxy that proxy-mode=ipvs, please.

**3. Load ipvs required kernel modules**

The following kernel modules are required by IPVS-based kube-proxy:

```shell
ip_vs
ip_vs_rr
ip_vs_wrr
ip_vs_sh
nf_conntrack_ipv4
```

IPVS-based kube-proxy will load them automatically. If it fails to load them, please check whether they are compiled into your kernel.
