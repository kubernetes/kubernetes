How to create a k8s cluster on Arm platforms by using kubeadm
=============================================================

Introduction
============

Kubernetes (K8s) is a commonly used Container Orchestration Engine. The first
step to using it is to setup the cluster, which is rather complicated. So this
article aims to provide user guide to deploy a K8s cluster on Arm64 servers
using a tool called **kubeadm.** This guide will deploy a K8s cluster which is
based on Arm64 platform.

Prerequisites
=============

-   The audience has a basic understanding of Kubernetes and Docker technology.

-   These instructions are assuming Ubuntu 16.04 or later is being used.

Deployment
==========

Now let's set up and using a Docker cluster of Kubernetes running on Arm64.

01, Basic information about Environment & k8s Components
--------------------------------------------------------

-   Cluster info

>   **Master nodes**: 1 Arm64 server (single mode) or 3 Arm64 servers (HA mode:
>   3 masters, 3 etcd servers).

>   In this document, we only introduce single mode for easier deployment.

>   Suppose it to be:

>          **10.20.30.1**

>   **Minion nodes**: n Arm64 servers (1 \<= n \<= 100+）

-   ubuntu 16.04+

-   Kubernetes 1.7.5

-   Docker \>= 1.12

-   Etcd \>= 3.0

-   CNI: Flanneld, Weaver ...

-   TLS

-   RBAC

-   kubelet TLS BootStrapping

-   Required addons in this document: kubedns dashboard

-   Suggested addons in production environment, but not touched in this
    document: heapster influxdb grafana EFK(elasticsearch, fluentd, kibana)
    Prometheus docker registry ceph

02, Install docker on each nodes and masters
--------------------------------------------

-   *apt-get -y install docker.io*

03, Disable swap & install kubelet/kubeadm/kubectl in each nodes and masters
----------------------------------------------------------------------------

**      Disable swap on your machines:**

1.  Identify configured swap devices and files with cat /proc/swaps.

2.  Turn off all swap devices and files with swapoff -a.

3.  Remove any matching reference found in /etc/fstab.

**       For each machine:**

-   *apt-get update && apt-get install -y apt-transport-https*

-   *curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg \| apt-key
    add -*

-   *cat \<\<EOF \>/etc/apt/sources.list.d/kubernetes.list*

-   *deb http://apt.kubernetes.io/ kubernetes-xenial main*

-   *EOF*

-   *apt-get update*

-   *apt install kubelet kubeadm kubectl*

04, Initializing master
-----------------------

**Perform the following steps with root privileges to deploy the master:**

* mkdir -p /etc/kubeadm*

*cat \<\<EOF \> /etc/kubeadm/kubeadm-init.yaml*

*apiVersion: kubeadm.k8s.io/v1alpha1*

*kind: MasterConfiguration*

*kubernetesVersion: v1.7.5*

*selfHosted: true*

*authorizationMode: RBAC*

*networking:*

*dnsDomain: cluster.local*

*serviceSubnet: 10.96.0.0/12*

*podSubnet: 10.244.0.0/16*

*EOF*

* kubeadm init --config=/etc/kubeadm/kubeadm-init.yaml*

**          The output should be look like:        **

*.... .... To start using your cluster, you need to run (as a regular user):*

*mkdir -p \$HOME/.kube sudo cp -i /etc/kubernetes/admin.conf \$HOME/.kube/config
sudo chown \$(id -u):\$(id -g) \$HOME/.kube/config*

*You should now deploy a pod network to the cluster. Run "kubectl apply -f
[podnetwork].yaml" with one of the options listed at:*
<http://kubernetes.io/docs/admin/addons/>

*You can now join any number of machines by running the following on each node
as root:*

*kubeadm join --token \<token\> \<master-ip\>:\<master-port\>*

*Make a record of the cluster enabled command and kubeadm join command
that kubeadm init outputs. You will need this in a moment.*

05, Enable kubectl in master
----------------------------

**Perform the following steps to deploy the etcd cluster:**

*sudo mkdir -p \$HOME/.kube*

*sudo cp -i /etc/kubernetes/admin.conf \$HOME/.kube/config*

*sudo chown \$(id -u):\$(id -g) \$HOME/.kube/config*

06, Installing a pod network
----------------------------

**Please use the files in same folder, use commands as following:**

*kubectl apply -f kube-flannel-rbac.yml*

*kubectl apply -f kube-flannel.yml*

07, Joining your nodes
----------------------

**Run the command that was output by kubeadm init. For example:  **

>   *kubeadm join --token \<token\> \<master-ip\>:\<master-port\>*

08, How to tear down
--------------------

To undo what kubeadm did, you should first [drain the
node](https://kubernetes.io/docs/user-guide/kubectl/v1.6/#drain) and make sure
that the node is empty before shutting it down.

 Talking to the master with the appropriate credentials, run:

*kubectl drain \<node name\> --delete-local-data --force --ignore-daemonsets*

*kubectl delete node \<node name\>*

Then, on the node being removed, reset all kubeadm installed state:

*kubeadm reset*

09, Check NodePort & Expose Dashboard Service
---------------------------------------------

**Please use the files in same folder, and use command as following:**

* kubectl create -f kube-dashboard.yml*

In this file, I enabled 'NodePort' for it, so you can access the dashboard by
external networking.

Please check the result:

*\$ kubectl get svc -n kube-system \| grep dashboard*

*NAME CLUSTER-IP EXTERNAL-IP PORT(S) AGE*

*kubernetes-dashboard 10.99.183.135 \<nodes\> 80:30390/TCP 25s*

*\$ kubectl get svc -n kube-system -o yaml \| grep nodePort*

*- nodePort: 30390*

Let's check the dashboard, I have exposed NodePort, so we can access the
dashboard with URL: **http://nodeIP:nodePort**

For example, for this case, the value of 'nodeIP' is the IP of any minion's IP,
and the value of 'nodePort' is 30390.

So, the URL should be <http://10.20.30.xx:30390>
