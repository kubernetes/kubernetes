- [IPVS](#ipvs)
  - [What is IPVS](#what-is-ipvs)
  - [IPVS vs. IPTABLES](#ipvs-vs-iptables)
    - [When ipvs falls back to iptables](#when-ipvs-falls-back-to-iptables)
  - [Run kube-proxy in ipvs mode](#run-kube-proxy-in-ipvs-mode)
    - [Prerequisite](#prerequisite)
    - [Local UP Cluster](#local-up-cluster)
    - [GCE Cluster](#gce-cluster)
    - [Cluster Created by Kubeadm](#cluster-created-by-kubeadm)
  - [Debug](#debug)
    - [Check IPVS proxy rules](#check-ipvs-proxy-rules)
    - [Why kube-proxy can't start IPVS mode](#why-kube-proxy-cant-start-ipvs-mode)

# IPVS

This document intends to show users
- what is IPVS
- difference between IPVS and IPTABLES
- how to run kube-proxy in ipvs mode and info on debugging

## What is IPVS

**IPVS (IP Virtual Server)** implements transport-layer load balancing, usually called Layer 4 LAN switching, as part of
Linux kernel.

IPVS runs on a host and acts as a load balancer in front of a cluster of real servers. IPVS can direct requests for TCP
and UDP-based services to the real servers, and make services of real servers appear as virtual services on a single IP address.

## IPVS vs. IPTABLES
IPVS mode was introduced in Kubernetes v1.8 and goes beta in v1.9. IPTABLES mode was added in v1.1 and become the default operating mode since v1.2. Both IPVS and IPTABLES are based on `netfilter`.
Differences between IPVS mode and IPTABLES mode are as follows:

1. IPVS provides better scalability and performance for large clusters. 

2. IPVS supports more sophisticated load balancing algorithms than iptables (least load, least connections, locality, weighted, etc.).  

3. IPVS supports server health checking and connection retries, etc.
 
### When ipvs falls back to iptables
IPVS proxier will employ iptables in doing packet filtering, SNAT and supporting NodePort type service. Specifically, ipvs proxier will fall back on iptables in the following 4 scenarios.

**1. kube-proxy starts with --masquerade-all=true**

If kube-proxy starts with `--masquerade-all=true`, ipvs proxier will masquerade all traffic accessing service Cluster IP, which behaves the same as what iptables proxier. Suppose there is a service with Cluster IP `10.244.5.1` and port `8080`, then the iptables installed by ipvs proxier should be like what is shown below.

```shell
# iptables -t nat -nL

Chain PREROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain POSTROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-POSTROUTING  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes postrouting rules */

Chain KUBE-POSTROUTING (1 references)
target     prot opt source               destination         
MASQUERADE  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service traffic requiring SNAT */ mark match 0x4000/0x4000

Chain KUBE-MARK-DROP (0 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x8000

Chain KUBE-MARK-MASQ (6 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x4000

Chain KUBE-SERVICES (2 references)
target     prot opt source               destination         
KUBE-MARK-MASQ  tcp  -- 0.0.0.0/0        10.244.5.1            /* default/foo:http cluster IP */ tcp dpt:8080
```

**2. Specify cluster CIDR in kube-proxy startup**

If kube-proxy starts with `--cluster-cidr=<cidr>`, ipvs proxier will masquerade off-cluster traffic accessing service Cluster IP, which behaves the same as what iptables proxier. Suppose kube-proxy is provided with the cluster cidr `10.244.16.0/24`, and service Cluster IP is `10.244.5.1` and port is `8080`, then the iptables installed by ipvs proxier should be like what is shown below.

```shell
# iptables -t nat -nL

Chain PREROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain POSTROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-POSTROUTING  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes postrouting rules */

Chain KUBE-POSTROUTING (1 references)
target     prot opt source               destination         
MASQUERADE  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service traffic requiring SNAT */ mark match 0x4000/0x4000

Chain KUBE-MARK-DROP (0 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x8000

Chain KUBE-MARK-MASQ (6 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x4000

Chain KUBE-SERVICES (2 references)
target     prot opt source               destination         
KUBE-MARK-MASQ  tcp  -- !10.244.16.0/24        10.244.5.1            /* default/foo:http cluster IP */ tcp dpt:8080
```

**3. Load Balancer Source Ranges is specified for LB type service**

When service's `LoadBalancerStatus.ingress.IP` is not empty and service's `LoadBalancerSourceRanges` is specified, ipvs proxier will install iptables which looks like what is shown below. 

Suppose service's `LoadBalancerStatus.ingress.IP` is `10.96.1.2` and service's `LoadBalancerSourceRanges` is `10.120.2.0/24`.

```shell
# iptables -t nat -nL

Chain PREROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain POSTROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-POSTROUTING  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes postrouting rules */

Chain KUBE-POSTROUTING (1 references)
target     prot opt source               destination         
MASQUERADE  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service traffic requiring SNAT */ mark match 0x4000/0x4000

Chain KUBE-MARK-DROP (0 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x8000

Chain KUBE-MARK-MASQ (6 references)
target     prot opt source               destination         
MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x4000

Chain KUBE-SERVICES (2 references)
target     prot opt source       destination         
ACCEPT  tcp  -- 10.120.2.0/24    10.96.1.2       /* default/foo:http loadbalancer IP */ tcp dpt:8080
DROP    tcp  -- 0.0.0.0/0        10.96.1.2       /* default/foo:http loadbalancer IP */ tcp dpt:8080
```

**4. Support NodePort type service**

For supporting NodePort type service, ipvs will recruit the existing implementation in iptables proxier. For example, 

```shell
# kubectl describe svc nginx-service
Name:			nginx-service
...
Type:			NodePort
IP:			    10.101.28.148
Port:			http	3080/TCP
NodePort:		http	31604/TCP
Endpoints:		172.17.0.2:80
Session Affinity:	None

# iptables -t nat -nL

[root@100-106-179-225 ~]# iptables -t nat -nL
Chain PREROUTING (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination         
KUBE-SERVICES  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service portals */

Chain KUBE-SERVICES (2 references)
target     prot opt source               destination         
KUBE-MARK-MASQ  tcp  -- !172.16.0.0/16        10.101.28.148        /* default/nginx-service:http cluster IP */ tcp dpt:3080
KUBE-SVC-6IM33IEVEEV7U3GP  tcp  --  0.0.0.0/0            10.101.28.148        /* default/nginx-service:http cluster IP */ tcp dpt:3080
KUBE-NODEPORTS  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service nodeports; NOTE: this must be the last rule in this chain */ ADDRTYPE match dst-type LOCAL

Chain KUBE-NODEPORTS (1 references)
target     prot opt source               destination         
KUBE-MARK-MASQ  tcp  --  0.0.0.0/0            0.0.0.0/0            /* default/nginx-service:http */ tcp dpt:31604
KUBE-SVC-6IM33IEVEEV7U3GP  tcp  --  0.0.0.0/0            0.0.0.0/0            /* default/nginx-service:http */ tcp dpt:31604

Chain KUBE-SVC-6IM33IEVEEV7U3GP (2 references)
target     prot opt source               destination
KUBE-SEP-Q3UCPZ54E6Q2R4UT  all  --  0.0.0.0/0            0.0.0.0/0            /* default/nginx-service:http */
Chain KUBE-SEP-Q3UCPZ54E6Q2R4UT (1 references)
target     prot opt source               destination         
KUBE-MARK-MASQ  all  --  172.17.0.2           0.0.0.0/0            /* default/nginx-service:http */
DNAT       tcp  --  0.0.0.0/0            0.0.0.0/0            /* default/nginx-service:http */ tcp to:172.17.0.2:80
```
## Run kube-proxy in ipvs mode

Currently, local-up scripts, GCE scripts and kubeadm support switching IPVS proxy mode via exporting environment variables or specifying flags.  

### Prerequisite
Ensure IPVS required kernel modules
```shell
ip_vs
ip_vs_rr
ip_vs_wrr
ip_vs_sh
nf_conntrack_ipv4
```
1. have been compiled into the node kernel. Use

`grep -e ipvs -e nf_conntrack_ipv4 /lib/modules/$(uname -r)/modules.builtin`

and get results like the followings if compiled into kernel.
```
kernel/net/ipv4/netfilter/nf_conntrack_ipv4.ko
kernel/net/netfilter/ipvs/ip_vs.ko
kernel/net/netfilter/ipvs/ip_vs_rr.ko
kernel/net/netfilter/ipvs/ip_vs_wrr.ko
kernel/net/netfilter/ipvs/ip_vs_lc.ko
kernel/net/netfilter/ipvs/ip_vs_wlc.ko
kernel/net/netfilter/ipvs/ip_vs_fo.ko
kernel/net/netfilter/ipvs/ip_vs_ovf.ko
kernel/net/netfilter/ipvs/ip_vs_lblc.ko
kernel/net/netfilter/ipvs/ip_vs_lblcr.ko
kernel/net/netfilter/ipvs/ip_vs_dh.ko
kernel/net/netfilter/ipvs/ip_vs_sh.ko
kernel/net/netfilter/ipvs/ip_vs_sed.ko
kernel/net/netfilter/ipvs/ip_vs_nq.ko
kernel/net/netfilter/ipvs/ip_vs_ftp.ko
```

OR

2. have been loaded.
```shell
# load module <module_name>
modprobe -- ip_vs
modprobe -- ip_vs_rr
modprobe -- ip_vs_wrr
modprobe -- ip_vs_sh
modprobe -- nf_conntrack_ipv4

# to check loaded modules, use
lsmod | grep -e ipvs -e nf_conntrack_ipv4
# or
cut -f1 -d " "  /proc/modules | grep -e ip_vs -e nf_conntrack_ipv4
 ```

Packages such as `ipset` should also be installed on the node before using IPVS mode.  

Kube-proxy will fall back to IPTABLES mode if those requirements are not met.

### Local UP Cluster

Kube-proxy will run in iptables mode by default in a [local-up cluster](https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md). 

To use IPVS mode, users should export the env `KUBE_PROXY_MODE=ipvs` to specify the ipvs mode before [starting the cluster](https://github.com/kubernetes/community/blob/master/contributors/devel/running-locally.md#starting-the-cluster):
```shell
# before running `hack/local-up-cluster.sh`
export KUBE_PROXY_MODE=ipvs
```

### GCE Cluster

Similar to local-up cluster, kube-proxy in [clusters running on GCE](https://kubernetes.io/docs/getting-started-guides/gce/) run in iptables mode by default. Users need to  export the env `KUBE_PROXY_MODE=ipvs` before [starting a cluster](https://kubernetes.io/docs/getting-started-guides/gce/#starting-a-cluster):
```shell
#before running one of the commmands chosen to start a cluster:
# curl -sS https://get.k8s.io | bash
# wget -q -O - https://get.k8s.io | bash
# cluster/kube-up.sh
export KUBE_PROXY_MODE=ipvs
```

### Cluster Created by Kubeadm

Kube-proxy will run in iptables mode by default in a cluster deployed by [kubeadm](https://kubernetes.io/docs/setup/independent/create-cluster-kubeadm/). 

If you are using kubeadm with a [configuration file](https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-init/#config-file), you can specify the ipvs mode adding `SupportIPVSProxyMode: true` below the `kubeProxy` field.

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
before running

`kube init --config <path_to_configuration_file>`

If you are using Kubernetes v1.8, you can also add the flag `--feature-gates=SupportIPVSProxyMode=true` (deprecated since v1.9) in `kubeadm init` command

```
kubeadm init --feature-gates=SupportIPVSProxyMode=true
```

to specify the ipvs mode before deploying the cluster.

**Notes**  
If ipvs mode is successfully on, you should see ipvs proxy rules (use `ipvsadm`) like
```shell
 # ipvsadm -ln
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
TCP  10.0.0.1:443 rr persistent 10800
  -> 192.168.0.1:6443             Masq    1      1          0
```
or similar logs occur in kube-proxy logs (for example, `/tmp/kube-proxy.log` for local-up cluster)  when the local cluster is running:
```
Using ipvs Proxier.
```

While there is no ipvs proxy rules or the following logs ocuurs indicate that the kube-proxy fails to use ipvs mode: 
```
Can't use ipvs proxier, trying iptables proxier
Using iptables Proxier.
```
See the following section for more details on debugging.

## Debug

### Check IPVS proxy rules

Users can use `ipvsadm` tool to check whether kube-proxy are maintaining IPVS rules correctly. For example, we have the following services in the cluster:

```
 # kubectl get svc --all-namespaces
NAMESPACE     NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)         AGE
default       kubernetes   ClusterIP   10.0.0.1     <none>        443/TCP         1d
kube-system   kube-dns     ClusterIP   10.0.0.10    <none>        53/UDP,53/TCP   1d
```
We may get IPVS proxy rules like:

```shell
 # ipvsadm -ln
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
TCP  10.0.0.1:443 rr persistent 10800
  -> 192.168.0.1:6443             Masq    1      1          0
TCP  10.0.0.10:53 rr
  -> 172.17.0.2:53                Masq    1      0          0
UDP  10.0.0.10:53 rr
  -> 172.17.0.2:53                Masq    1      0          0
```

### Why kube-proxy can't start IPVS mode

Use the following check list to help you solve the problems: 

**1. Enable IPVS feature gateway**

For Kubernetes v1.10 and later, feature gate `SupportIPVSProxyMode` is set to `true` by default. However, you need to enable `--feature-gates=SupportIPVSProxyMode=true` explicitly for Kubernetes before v1.10.

**2. Specify proxy-mode=ipvs**

Check whether the kube-proxy mode has been set to `ipvs`.

**3. Install required kernel modules and packages**

Check whether the ipvs required kernel modules have been compiled into the kernel and packages installed. (see Prerequisite)
