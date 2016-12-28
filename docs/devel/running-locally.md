Getting started locally
-----------------------

**Table of Contents**

- [Requirements](#requirements)
    - [Linux](#linux)
    - [Docker](#docker)
    - [etcd](#etcd)
    - [go](#go)
    - [OpenSSL](#openssl)
- [Clone the repository](#clone-the-repository)
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

Not running Linux? Consider running [Minikube](http://kubernetes.io/docs/getting-started-guides/minikube/), or on a cloud provider like [Google Compute Engine](../getting-started-guides/gce.md).

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation)
1.3+. Ensure the Docker daemon is running and can be contacted (try `docker
ps`).  Some of the Kubernetes components need to run as root, which normally
works fine with docker.

#### etcd

You need an [etcd](https://github.com/coreos/etcd/releases) in your path, please make sure it is installed and in your ``$PATH``.

#### go

You need [go](https://golang.org/doc/install) in your path (see [here](development.md#go-versions) for supported versions), please make sure it is installed and in your ``$PATH``.

#### OpenSSL

You need [OpenSSL](https://www.openssl.org/) installed.  If you do not have the `openssl` command available, you may see the following error in `/tmp/kube-apiserver.log`:

```
server.go:333] Invalid Authentication Config: open /tmp/kube-serviceaccount.key: no such file or directory
```

### Clone the repository

In order to run kubernetes you must have the kubernetes code on the local machine. Cloning this repository is sufficient.

```$ git clone --depth=1 https://github.com/kubernetes/kubernetes.git```

The `--depth=1` parameter is optional and will ensure a smaller download.

### Starting the cluster

In a separate tab of your terminal, run the following (since one needs sudo access to start/stop Kubernetes daemons, it is easier to run the entire script as root):

```sh
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master
and a single node. Type Control-C to shut it down.

If you've already compiled the Kubernetes components, then you can avoid rebuilding them with this script by using the `-O` flag.

```sh
./hack/local-up-cluster.sh -O
```

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
cluster/kubectl.sh create -f test/fixtures/doc-yaml/user-guide/pod.yaml
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
make
hack/local-up-cluster.sh
```

#### kubectl claims to start a container but `get pods` and `docker ps` don't show it.

One or more of the Kubernetes daemons might've crashed. Tail the logs of each in /tmp.

#### The pods fail to connect to the services by host names

To start the DNS service, you need to set the following variables:

```sh
KUBE_ENABLE_CLUSTER_DNS=true
KUBE_DNS_SERVER_IP="10.0.0.10"
KUBE_DNS_DOMAIN="cluster.local"
KUBE_DNS_REPLICAS=1
```

To know more on DNS service you can look [here](http://issue.k8s.io/6667). Related documents can be found [here](../../build/kube-dns/#how-do-i-configure-it)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/running-locally.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
