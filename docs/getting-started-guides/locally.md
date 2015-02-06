## Getting started locally

### Requirements

#### Linux

Not running Linux? Consider running Linux in a local virtual machine with [Vagrant](vagrant.md), or on a cloud provider like [Google Compute Engine](gce.md)

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation) 1.0.0+. Ensure the Docker daemon is running and can be contacted (try `docker ps`).  Some of the kubernetes components need to run as root, which normally works fine with docker.

#### etcd

You need an [etcd](https://github.com/coreos/etcd/releases/tag/v0.4.6) in your path, please make sure it is installed and in your ``$PATH``.

#### go

You need [go](https://golang.org/doc/install) in your path, please make sure it is installed and in your ``$PATH``.

### Starting the cluster

In a separate tab of your terminal, run:

```
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master
and a single minion. Type Control-C to shut it down.

You can use the cluster/kubectl.sh script to interact with the local cluster.
You must set the KUBERNETES_PROVIDER and KUBERNETES_MASTER environment variables to let other programs
know how to reach your master.

```
export KUBERNETES_PROVIDER=local
export KUBERNETES_MASTER=http://localhost:8080
```

### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kubectl.sh commands to interact with your local setup.

```
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationControllers
cluster/kubectl.sh run-container my-nginx --image=dockerfile/nginx --replicas=2 --port=80


## begin wait for provision to complete, you can monitor the docker pull by opening a new terminal
  sudo docker images
  ## you should see it pulling the dockerfile/nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## introspect kubernetes!
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationControllers
```

Congratulations!

### Troubleshooting

#### I can't reach service IPs on the network.

Some firewall software that uses iptables may not interact well with
kubernetes.  If you're having trouble around networking, try disabling any
firewall or other iptables-using systems, first.

By default the IP range for service portals is 10.0.*.* - depending on your
docker installation, this may conflict with IPs for containers.  If you find
containers running with IPs in this range, edit hack/local-cluster-up.sh and
change the portal_net flag to something else.

#### I cannot create a replication controller with replica size greater than 1!  What gives?

You are running a single minion setup.  This has the limitation of only supporting a single replica of a given pod.  If you are interested in running with larger replica sizes, we encourage you to try the local vagrant setup or one of the cloud providers.

#### I changed Kubernetes code, how do I run it?

```
cd kubernetes
hack/build-go.sh
hack/local-up-cluster.sh
```
