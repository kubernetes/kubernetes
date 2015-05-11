## Getting started locally

### Requirements

#### Linux

Not running Linux? Consider running Linux in a local virtual machine with [Vagrant](vagrant.md), or on a cloud provider like [Google Compute Engine](gce.md)

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation)
1.3+. Ensure the Docker daemon is running and can be contacted (try `docker
ps`).  Some of the kubernetes components need to run as root, which normally
works fine with docker.

#### etcd

You need an [etcd](https://github.com/coreos/etcd/releases) in your path, please make sure it is installed and in your ``$PATH``.

#### go

You need [go](https://golang.org/doc/install) at least 1.3+ in your path, please make sure it is installed and in your ``$PATH``.

### Starting the cluster

In a separate tab of your terminal, run the following (since one needs sudo access to start/stop kubernetes daemons, it is easier to run the entire script as root):

```
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master
and a single minion. Type Control-C to shut it down.

You can use the cluster/kubectl.sh script to interact with the local cluster. hack/local-up-cluster.sh will
print the commands to run to point kubectl at the local cluster.


### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kubectl.sh commands to interact with your local setup.

```
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationControllers
cluster/kubectl.sh run-container my-nginx --image=nginx --replicas=2 --port=80


## begin wait for provision to complete, you can monitor the docker pull by opening a new terminal
  sudo docker images
  ## you should see it pulling the nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## introspect kubernetes!
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get replicationControllers
```


### Running a user defined pod

Note the difference between a [container](http://docs.k8s.io/containers.md)
and a [pod](http://docs.k8s.io/pods.md). Since you only asked for the former, kubernetes will create a wrapper pod for you.
However you can't view the nginx start page on localhost. To verify that nginx is running you need to run `curl` within the docker container (try `docker exec`).

You can control the specifications of a pod via a user defined manifest, and reach nginx through your browser on the port specified therein:

```
cluster/kubectl.sh create -f examples/pod.yaml
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

#### kubectl claims to start a container but `get pods` and `docker ps` don't show it.

One or more of the kubernetes daemons might've crashed. Tail the logs of each in /tmp.

#### The pods fail to connect to the services by host names
The local-up-cluster.sh script doesn't start a DNS service. Similar situation can be found [here](https://github.com/GoogleCloudPlatform/kubernetes/issues/6667). You can start a manually. Related documents can be found [here](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/cluster/addons/dns#how-do-i-configure-it)
