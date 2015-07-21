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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/locally.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started locally
-----------------------

**Table of Contents**

- [Requirements](#requirements)
    - [Linux](#linux)
    - [Docker](#docker)
    - [etcd](#etcd)
    - [go](#go)
- [Starting the cluster](#starting-the-cluster)
- [Running a container](#running-a-container)
- [Running a user defined pod](#running-a-user-defined-pod)
- [Troubleshooting](#troubleshooting)
    - [I cannot reach service IPs on the network.](#i-cannot-reach-service-ips-on-the-network)
    - [I cannot create a replication controller with replica size greater than 1!  What gives?](#i-cannot-create-a-replication-controller-with-replica-size-greater-than-1--what-gives)
    - [I changed Kubernetes code, how do I run it?](#i-changed-kubernetes-code-how-do-i-run-it)
    - [kubectl claims to start a container but `get pods` and `docker ps` don't show it.](#kubectl-claims-to-start-a-container-but-get-pods-and-docker-ps-dont-show-it)
    - [The pods fail to connect to the services by host names](#the-pods-fail-to-connect-to-the-services-by-host-names)

### Requirements

#### Linux

Not running Linux? Consider running Linux in a local virtual machine with [Vagrant](vagrant.md), or on a cloud provider like [Google Compute Engine](gce.md)

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation)
1.3+. Ensure the Docker daemon is running and can be contacted (try `docker
ps`).  Some of the Kubernetes components need to run as root, which normally
works fine with docker.

#### etcd

You need an [etcd](https://github.com/coreos/etcd/releases) in your path, please make sure it is installed and in your ``$PATH``.

#### go

You need [go](https://golang.org/doc/install) at least 1.3+ in your path, please make sure it is installed and in your ``$PATH``.

### Starting the cluster

In a separate tab of your terminal, run the following (since one needs sudo access to start/stop Kubernetes daemons, it is easier to run the entire script as root):

```sh
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master
and a single node. Type Control-C to shut it down.

You can use the cluster/kubectl.sh script to interact with the local cluster. hack/local-up-cluster.sh will
print the commands to run to point kubectl at the local cluster.


### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kubectl.sh commands to interact with your local setup.

```sh
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationcontrollers
cluster/kubectl.sh run my-nginx --image=nginx --replicas=2 --port=80


## begin wait for provision to complete, you can monitor the docker pull by opening a new terminal
  sudo docker images
  ## you should see it pulling the nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## introspect Kubernetes!
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationcontrollers
```


### Running a user defined pod

Note the difference between a [container](../user-guide/containers.md)
and a [pod](../user-guide/pods.md). Since you only asked for the former, Kubernetes will create a wrapper pod for you.
However you cannot view the nginx start page on localhost. To verify that nginx is running you need to run `curl` within the docker container (try `docker exec`).

You can control the specifications of a pod via a user defined manifest, and reach nginx through your browser on the port specified therein:

```sh
cluster/kubectl.sh create -f docs/user-guide/pod.yaml
```

Congratulations!

### Troubleshooting

#### I cannot reach service IPs on the network.

Some firewall software that uses iptables may not interact well with
kubernetes.  If you have trouble around networking, try disabling any
firewall or other iptables-using systems, first.  Also, you can check
if SELinux is blocking anything by running a command such as `journalctl --since yesterday | grep avc`.

By default the IP range for service cluster IPs is 10.0.*.* - depending on your
docker installation, this may conflict with IPs for containers.  If you find
containers running with IPs in this range, edit hack/local-cluster-up.sh and
change the service-cluster-ip-range flag to something else.

#### I cannot create a replication controller with replica size greater than 1!  What gives?

You are running a single node setup.  This has the limitation of only supporting a single replica of a given pod.  If you are interested in running with larger replica sizes, we encourage you to try the local vagrant setup or one of the cloud providers.

#### I changed Kubernetes code, how do I run it?

```sh
cd kubernetes
hack/build-go.sh
hack/local-up-cluster.sh
```

#### kubectl claims to start a container but `get pods` and `docker ps` don't show it.

One or more of the KUbernetes daemons might've crashed. Tail the logs of each in /tmp.

#### The pods fail to connect to the services by host names

The local-up-cluster.sh script doesn't start a DNS service. Similar situation can be found [here](https://github.com/GoogleCloudPlatform/kubernetes/issues/6667). You can start a manually. Related documents can be found [here](../../cluster/addons/dns/#how-do-i-configure-it)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/locally.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
