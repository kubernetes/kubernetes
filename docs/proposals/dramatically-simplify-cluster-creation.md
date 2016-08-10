# Proposal: Dramatically Simplify Kubernetes Cluster Creation

Luke Marsden & many others in [SIG-cluster-lifecycle](https://github.com/kubernetes/community/tree/master/sig-cluster-lifecycle).

10th August 2016

*This proposal aims to capture the latest consensus and proposed plan of action of SIG-cluster-lifecycle. It should satisfy the first bullet point [required by the feature description](https://github.com/kubernetes/features/issues/11).*

See also: [this presentation to community hangout on 4th August 2016](https://docs.google.com/presentation/d/17xrFxrTwqrK-MJk0f2XCjfUPagljG7togXHcC39p0sM/edit?ts=57a33e24#slide=id.g158d2ee41a_0_76)

## Motivation

Kubernetes is hard to install, and there are many different ways to do it today. None of them are excellent. We believe this is hindering adoption.

## Goals

Have one recommended, official, tested, "happy path" which will enable a majority of new and existing Kubernetes users to:

* Kick the tires and easily turn up a new cluster on infrastructure of their choice

* Get a reasonably secure, production-ready cluster, with reasonable defaults and a range of easily-installable add-ons

We plan to do so by improving and simplifying Kubernetes itself, rather than building lots of tooling which "wraps" Kubernetes by poking all the bits into the right place.

## Scope of project

There are logically 3 steps to deploying a Kubernetes cluster:

1. *Provisioning*: Getting some servers - these may be VMs on a developer's workstation, VMs in public clouds, or bare-metal servers in a user's data center.

2. *Install & Discovery*: Installing the Kubernetes core components on those servers (kubelet, etc) - and bootstrapping the cluster to a state of basic liveness, including allowing each server in the cluster to discover other servers: for example teaching etcd servers about their peers, having TLS certificates provisioned, etc.

3. *Add-ons*: Now that basic cluster functionality is working, installing add-ons such as DNS or a pod network (should be possible using kubectl apply).

Notably, this project is *only* working on dramatically improving 2 and 3 from the perspective of users typing commands directly into root shells of servers. The reason for this is that there are a great many different ways of provisioning servers, and users will already have their own preferences.

What's more, once we've radically improved the user experience of 2 and 3, it will make the job of tools that want to do all three much easier.

## User stories

### Phase I

**_In time to be an alpha feature in Kubernetes 1.4._**

* *Install*: As a potential Kubernetes user, I can deploy a Kubernetes 1.4 cluster on a handful of computers running Linux and Docker by typing two commands on each of those computers. The process is so simple that it becomes obvious to me how to easily automate it if I so wish.

* *Pre-flight check*: If any of the computers don't have working dependencies installed (e.g. bad version of Docker, too-old Linux kernel), I am informed early on and given clear instructions on how to fix it so that I can keep trying until it works.

* *Control*: Having provisioned a cluster, I can gain user credentials which allow me to remotely control it using kubectl.

* *Install-addons*: I can select from a set of recommended add-ons to install directly after installing Kubernetes on my set of initial computers with kubectl apply.

* *Add-node*: I can add another computer to the cluster.

* *Secure*: As an attacker with (presumed) control of the network, I cannot add malicious nodes I control to the cluster created by the user. I also cannot remotely control the cluster.

### Phase II

**_In time for Kubernetes 1.5:_**
*Everything from Phase I as beta/stable feature, everything else below as beta feature in Kubernetes 1.5.*

* *Upgrade*: Later, when Kubernetes 1.4.1 or any newer release is published, I can upgrade to it by typing one other command on each computer.

* *HA*: If one of the computers in the cluster fails, the cluster carries on working. I can find out how to replace the failed computer, including if the computer was one of the masters.

## Top-down view: UX for Phase I items

We will introduce a new binary, kubeadm, which ships with the Kubernetes OS packages (and binary tarballs, for OSes without package managers).

```
laptop$ kubeadm --help

kubeadm: bootstrap a kubernetes cluster.

Global options:

    --discovery    Discovery mechanism to use. Choose from: gossip,
                   out-of-band. Default: gossip.

Subcommands:
    init master    Run this on the first server you deploy onto.

    join node      Run this on other servers to join an existing cluster.

    user create    Create user and fetch authentication config for them.
```

### Install

*On first machine:*

```
master# kubeadm init master
Initializing kubernetes master...  [done]
Cluster token: 73R2SIPM739TNZOA
Run the following command on machines you want to become nodes:
  kubeadm join node --token=73R2SIPM739TNZOA <master-ip>
You can now run kubectl here.
```

*On N "node" machines:*

```
node# kubeadm join node --token=73R2SIPM739TNZOA <master-ip>
Initializing kubernetes node...    [done]
Bootstrapping certificates...      [done]
Joined node to cluster, see 'kubectl get nodes' on master.
```

Note `[done]` would be colored green in all of the above.

### Install: alternative for automated deploy

*The user (or their config management system) creates a token and passes the same one to both init and join.*

```
master# kubeadm init master --token=73R2SIPM739TNZOA
Initializing kubernetes master...  [done]
You can now run kubectl here.
```

### Pre-flight check

```
master# kubeadm init master
Error: socat not installed. Unable to proceed.
```

### Control

*On master, after Install, kubectl is automatically able to talk to localhost:8080:*

```
master# kubectl get pods
[normal kubectl output]
```

*To mint new user credentials on the master:*

```
master# kubeadm user create -o kubeconfig-bob bob

Waiting for cluster to become ready...       [done]
Creating user certificate for user...        [done]
Waiting for user certificate to be signed... [done]
Your cluster configuration file has been saved in kubeconfig.

laptop# scp <master-ip>:/root/kubeconfig-bob ~/.kubeconfig
laptop# kubectl get pods
[normal kubectl output]
```

### Install-addons

*Using CNI network as example:*

```
master# kubectl apply --purge -f \
    https://git.io/kubernetes-addons/<X>.yaml
[normal kubectl apply output]
```

### Add-node

*Same as Install – "on node machines".*

### Secure

```
node# kubeadm join --token=GARBAGE node <master-ip>
Unable to join mesh network. Check your token.
```

## Work streams – critical path

1. [Debs](https://github.com/kubernetes/release/pull/35) and [RPMs](https://github.com/kubernetes/release/pull/50) (and binaries?) - so that kubernetes can be installed in the first place

    * Mike & Devan

2. [Cluster bootstrap with gossip](https://github.com/kubernetes/kubernetes/pull/30361) - so that we can implement the simple UX with no dependencies

    * Luke & Ilya

3. [TLS bootstrapping](https://github.com/kubernetes/features/issues/43) - so that the gossip bootstrap can provide ca-cert and address of API server to the TLS bootstrapping process, and have the kubelets bootstrap from there

    * Brandon, George Tankersley & Yifan Gu?

4. Documentation - so that new users can see this in 1.4 (even if it’s caveated with alpha/experimental labels and flags all over it)

    * Luke

### Nice to have

5. [Kubectl apply --purge](https://github.com/kubernetes/kubernetes/pull/29551) - so that addons can be maintained using k8s infrastructure

    * Luke & Ilya

6. [Dynamic Kubelet Settings](https://github.com/kubernetes/kubernetes/issues/27980) - so that kubeadm command can manage kubelet that is already running by rewriting its componentconfig and having it automatically reload its config

    * Michael Taufen

7. [Pre-flight checks](https://github.com/kubernetes/kubernetes/issues/26093) - so that kube(let) can bail if basic dependencies are not met

    * Lucas Käldström

## Prototype of UX (old)

Ilya and Luke at Weaveworks have prototyped the desired user experience listed described in the "Top-down view" section. This prototype is called kube-alpha for lack of a better name. It is currently living in [Ilya's fork of kubernetes-anywhere](https://github.com/errordeveloper/kubernetes-anywhere/tree/kube-alpha/phase2/kube-alpha).

Run dev-init.sh and then dev.sh on a machine with docker-machine installed and you'll get guided through the current prototype UX for installing a 2-node Kubernetes cluster on two Docker Machine VMs.

The purpose of this prototype is to demonstrate that the desired UX is possible and to enable discussion around it. The prototype is written in Golang because the final version will need to be integrated into the kubelet.

The techniques used in the prototype are not intended to be final!

Experimenting in this way allows us to start fleshing out the Golang interfaces that will make sense to provide for different implementations of e.g. discovery.

Ultimately, this project should aim to implement one or more implementations of these interfaces, including exactly one reasonable default.

## See also

* [Joe Beda's "K8s the hard way easier"](https://docs.google.com/document/d/1lJ26LmCP-I_zMuqs6uloTgAnHPcuT7kOYtQ7XSgYLMA/edit#heading=h.ilgrv18sg5t) which combines Kelsey's "Kubernetes the hard way" with history of proposed UX at the end (scroll all the way down to the bottom).
