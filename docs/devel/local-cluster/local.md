**Stop. This guide has been superseded by [Minikube](https://github.com/kubernetes/minikube) which is the recommended method of running Kubernetes on your local machine.**

### Requirements

#### Linux

Not running Linux? Consider running Linux in a local virtual machine with [vagrant](https://www.vagrantup.com/), or on a cloud provider like Google Compute Engine

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation)
1.8.3+. Ensure the Docker daemon is running and can be contacted (try `docker
ps`).  Some of the Kubernetes components need to run as root, which normally
works fine with docker.

#### etcd

You need an [etcd](https://github.com/coreos/etcd/releases) in your path, please make sure it is installed and in your ``$PATH``.

#### go

You need [go](https://golang.org/doc/install) at least 1.4+ in your path, please make sure it is installed and in your ``$PATH``.

### Starting the cluster

First, you need to [download Kubernetes](http://kubernetes.io/docs/getting-started-guides/binary_release/). Then open a separate tab of your terminal
and run the following (since one needs sudo access to start/stop Kubernetes daemons, it is easier to run the entire script as root):

```shell
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

```shell
export KUBERNETES_PROVIDER=local
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get deployments
cluster/kubectl.sh run my-nginx --image=nginx --replicas=2 --port=80

## begin wait for provision to complete, you can monitor the docker pull by opening a new terminal
  sudo docker images
  ## you should see it pulling the nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## create a service for nginx, which serves on port 80
cluster/kubectl.sh expose deployment my-nginx --port=80 --name=my-nginx

## introspect Kubernetes!
cluster/kubectl.sh get pods
cluster/kubectl.sh get services
cluster/kubectl.sh get deployments

## Test the nginx service with the IP/port from "get services" command
curl http://10.X.X.X:80/
```

### Running a user defined pod

Note the difference between a [container](http://kubernetes.io/docs/user-guide/containers/)
and a [pod](http://kubernetes.io/docs/user-guide/pods/). Since you only asked for the former, Kubernetes will create a wrapper pod for you.
However you cannot view the nginx start page on localhost. To verify that nginx is running you need to run `curl` within the docker container (try `docker exec`).

You can control the specifications of a pod via a user defined manifest, and reach nginx through your browser on the port specified therein:

```shell
cluster/kubectl.sh create -f test/fixtures/doc-yaml/user-guide/pod.yaml
```

Congratulations!

### FAQs

#### I cannot reach service IPs on the network.

Some firewall software that uses iptables may not interact well with
kubernetes.  If you have trouble around networking, try disabling any
firewall or other iptables-using systems, first.  Also, you can check
if SELinux is blocking anything by running a command such as `journalctl --since yesterday | grep avc`.

By default the IP range for service cluster IPs is 10.0.*.* - depending on your
docker installation, this may conflict with IPs for containers.  If you find
containers running with IPs in this range, edit hack/local-cluster-up.sh and
change the service-cluster-ip-range flag to something else.

#### I changed Kubernetes code, how do I run it?

```shell
cd kubernetes
hack/build-go.sh
hack/local-up-cluster.sh
```

#### kubectl claims to start a container but `get pods` and `docker ps` don't show it.

One or more of the Kubernetes daemons might've crashed. Tail the [logs](http://kubernetes.io/docs/admin/cluster-troubleshooting/#looking-at-logs) of each in /tmp.

```shell
$ ls /tmp/kube*.log
$ tail -f /tmp/kube-apiserver.log
```

#### The pods fail to connect to the services by host names

The local-up-cluster.sh script doesn't start a DNS service. Similar situation can be found [here](http://issue.k8s.io/6667). You can start a manually.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/local-cluster/local.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
