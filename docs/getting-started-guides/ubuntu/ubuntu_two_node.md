## Table of Contents

* [Introduction](#introduction)
* [Environment Setup](#environment-setup)
* [Component Installation](#component-installation)
* [Configure Kubernetes Hosts](#configure-kubernetes-hosts)
* [Configure Kubernetes Master](#configure-kubernetes-master)
* [Configure Kubernetes Node](#configure-kubernetes-node)
* [Conclusion](#conclusion)

## Introduction

This guide will take you through the manual process for getting started with a two-node Kubernetes cluster on Ubuntu. The setup will have one master and one minion to keep it simple.  This article is meant to get you familiar with the manual process for setting up Kubernetes. 

This guide is loosely based on a few of the other getting started guides. It also make use of the scripts provided in the project for setting up singe-server Ubuntu installs. 

## Environment Setup

You will need two servers setup within your chosen cloud environment. They will need network connectivity to each other and to the public Internet. 

We're using the below configuration values: 

    | Hostname | Private IP |
    | kube-master | 10.11.50.12 |
    | kube-minion | 10.11.50.11 | 

The first instance will be our Kubernetes master.

The second instance will be our first Kubernetes minion.

You will need to follow the steps on installing the pre-requisites and Kubernetes on both your master and minion instances. 

## Component Installation

Kubernetes has a few requirements, but not many and they're slowly being baked into the installation process. Ubuntu, however, has some pre-requisites it needs before you can move forward with installing and setting up Kubernetes.

You will want to run the following commands in the order laid out in this tutorial. These are broken into sections, but these can be combined into a single shell script if you're feeling energetic. 

### Install Docker

First, you need Docker on both the master and its nodes. These commands come from Docker's site and ensure you're working with the most recent version. By default, using `apt-get` to install docker will give you version 1.0.1, which is not ideal for what we want to do. Docker recommends running the following to ensure you have the latest version on your Ubuntu installation:

    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 36A1D7869245C8950F966E92D8576A8BA88D21E9
    sh -c "echo deb https://get.docker.com/ubuntu docker main > /etc/apt/sources.list.d/docker.list"
    apt-get update
    apt-get install -y lxc-docker git make
    source /etc/bash_completion.d/docker

We install `git` and `make` while we're at it, but that can be installed anywhere in the pre-requisite setup process. 

### Install *etcd 2.0*

Next, let's go ahead and install *etcd 2.0* on your master; *etcd* is not required on your Kubernetes nodes (minions). You could do this after installing Kubernetes if desired, but, personally, I treat this as a pre-requisite for Kubernetes and therefore it should come prior to installing and configuring Kubernetes. 

    curl -L  https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.tar.gz -o etcd-v2.0.0-linux-amd64.tar.gz
    tar xzvf etcd-v2.0.0-linux-amd64.tar.gz
    cd etcd-v2.0.0-linux-amd64
    mkdir /opt/bin
    cp etcd* /opt/bin

Now you're ready for Kubernetes. 

### Build Kubernetes

Do this on both your master and minion starting with the master. 

You can either install Kubernetes on Ubuntu using pre-existing binaries or build from source. Since Kubernetes is rapidly iterating we will be building from source in this tutorial which gives us all the latest patches and improvements.

You don't need to install Go since Kubernetes already has a Go build process when it starts its setup. This was a legacy requirement that has since been removed. 

You will want to perform the following commands on both the master and the minion:

    git clone https://github.com/GoogleCloudPlatform/kubernetes.git
    cd kubernetes
    make release

The build process creates the folder `_output` in the root of your `kubernetes` folder. The release is:

`_output/release-tars/kubernetes-server-linux-amd64.tar.gz`. 

Extract this archive to a folder of your choosing. You will be copying binaries out of `/server/bin` into `/opt/bin` on your master and nodes in the next few sections.

## Configure Kubernetes Hosts

You will now need to configure your *kube-master* and *kube-minion*. First, prepare the *hosts* file so that both can resolve each other via their hostnames. 

    echo "10.11.50.12 kube-master
    10.11.50.11  kube-minion" >> /etc/hosts

## Configure Kubernetes Master

Your *kube-master* will run the services: 

* etcd
* kube-apiserver
* kube-controller-manager
* kube-scheduler

Before configuring the master you will need to copy all configuration and binary files into their appropriate locations. You will need to extract the binaries from the release tar (noted above). This contains the `/server/bin` folder referenced in the next commands:

    cp server/bin/kube-apiserver /opt/bin/
    cp server/bin/kube-controller-manager /opt/bin/
    cp server/bin/kube-kube-scheduler /opt/bin/
    cp server/bin/kubecfg /opt/bin/
    cp server/bin/kubectl /opt/bin/
    cp server/bin/kubernetes /opt/bin/

Now, return to the directory you built Kubernetes: 

    cp kubernetes/cluster/ubuntu/init_conf/kube-apiserver.conf /etc/init/
    cp kubernetes/cluster/ubuntu/init_conf/kube-controller-manager.conf /etc/init/
    cp kubernetes/cluster/ubuntu/init_conf/kube-kube-scheduler.conf /etc/init/
    
    cp kubernetes/cluster/ubuntu/initd_scripts/kube-apiserver /etc/init.d/
    cp kubernetes/cluster/ubuntu/initd_scripts/kube-controller-manager /etc/init.d/
    cp kubernetes/cluster/ubuntu/initd_scripts/kube-kube-scheduler /etc/init.d/
    
    cp kubernetes/cluster/ubuntu/default_scripts/kubelet /etc/default/
    cp kubernetes/cluster/ubuntu/default_scripts/kube-proxy /etc/default/
    cp kubernetes/cluster/ubuntu/default_scripts/kubelet /etc/default/

### Configure the *etcd*

The default configuration should be updated to look like: 

    ETCD_OPTS="-listen-client-urls=http://kube-master:4001"

This should be done on *kube-master*. Your *kube-minion* will not be running *etcd*. 

### Configure *kube-apiserver*

On your *master* edit `/etc/default/kube-apiserver`. It should look something like this: 

    KUBE_APISERVER_OPTS="--address=0.0.0.0 \
    --port=8080 \
    --etcd_servers=http://kube-master:4001 \
    --portal_net=11.1.1.0/24 \
    --allow_privileged=false \
    --kubelet_port=10250 \
    --v=0"

### Configure *kube-controller-manager*

Configure the controller manager by editing `/etc/default/kube-controller-manager` to look like this: 

    KUBE_CONTROLLER_MANAGER_OPTS="--address=0.0.0.0 \
    --master=127.0.0.1:8080 \
    --machines=kube-minion \
    --v=0"

### Configure *kube-scheduler*

You will want to update the *kube-scheduler* configuration file to resemble this: 

    KUBE_SCHEDULER_OPTS="--address=0.0.0.0 \
    --master=127.0.0.1:8080 \
    --v=0"

Now, bring up the master. 

    service docker restart

### Validate Master Services

You can validate that the services are running on the *master* by running the command: 

    initctl list | grep -E '(kube|etc)'

You should see the services in a running state with a PID. If not, then the logs should be in `/var/log/upstart/`.

## Configure Kubernetes Node

Your *kube-minion* will run the services: 

* kubelet
* kube-proxy

First, start by copying the binaries each node will require: 

    cp server/bin/kubelet /opt/bin/
    cp server/bin/kube-proxy /opt/bin/
    cp server/bin/kubecfg /opt/bin/
    cp server/bin/kubectl /opt/bin/
    cp server/bin/kubernetes /opt/bin/

You will want to copy the following files from the Kubernetes project to their appropriate locations:

    cp kubernetes/cluster/ubuntu/init_conf/kubelet.conf /etc/init/
    cp kubernetes/cluster/ubuntu/init_conf/kube-proxy.conf /etc/init/

    cp kubernetes/cluster/ubuntu/initd_scripts/kubelet /etc/init.d/
    cp kubernetes/cluster/ubuntu/initd_scripts/kube-proxy /etc/init.d/

    cp kubernetes/cluster/ubuntu/default_scripts/kubelet /etc/default/
    cp kubernetes/cluster/ubuntu/default_scripts/kube-proxy /etc/default/

You will not be running *etcd* on *kube-minion*. Unfortunateyl, the init and configuration files in the Kubernetes */cluster/ubuntu* location were written for a single-server installation and are configured to start the Kubernetes services when the *etcd* service is started. This will require you to modify them. 

You will want to edit both `/etc/init/kubelet.conf` and `/etc/init/kube-proxy.conf`so that the lines: 

    start on started etcd
    stop on stopping etcd

are updated to look like this: 

    start on started docker
    stop on stopping docker

### Configure *kubelet*

Your *kubelet* configuration will need to look something like this: 

    KUBELET_OPTS="--address=0.0.0.0 \
    --port=10250 \
    --hostname_override=kube-minion \
    --etcd_servers=http://kube-master:4001 \
    --enable_server=true
    --v=0"

`--v=0` turns on debug. This is handy while you're learning way around, but you'll probably want to turn this down a bit before going into production. 

### Configure *kube-proxy*

Configure your proxy 

    KUBE_PROXY_OPTS="--etcd_servers=http://kube-master:4001 \
    --v=0"

Finally, restart services on your Kubernetes node:

    service docker restart

You can validate the service are running on the node by doing: 

    initctl list | grep -E '(docker|kube)'
    
They should both report as running with a PID. 

### Validate Node Services

You can validate that the services are running on the *node* by running the command: 

    initctl list | grep -E '(kube|docker)'

You should see the services in a running state with a PID. If not, then the logs should be in `/var/log/upstart/`. This applies to both the master and node. 

### View Nodes

You should also be able to view your nodes / minions. Run this from master: 

    /opt/bin/kubectl get minions
    
This should output:

    NAME                LABELS              STATUS
    kube-minion         <none>              Ready

## Conclusion

This tutorial should have you up and running with a single master and single minion setup on Ubuntu 14.04. We will cover how to setup multiple minions in a future article. So, stay tuned. 