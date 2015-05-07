## Getting started with Juju

Juju handles provisioning machines and deploying complex systems to a
wide number of clouds, supporting service orchestration once the bundle of
services has been deployed.



### Prerequisites

> Note: If you're running kube-up, on ubuntu - all of the dependencies
> will be handled for you. You may safely skip to the section:
> [Launch Kubernetes Cluster](#launch-kubernetes-cluster)

#### On Ubuntu

[Install the Juju client](https://juju.ubuntu.com/install) on your
local ubuntu system:

    sudo add-apt-repository ppa:juju/stable
    sudo apt-get update
    sudo apt-get install juju-core juju-quickstart


#### With Docker

If you are not using ubuntu or prefer the isolation of docker, you may
run the following:

    mkdir ~/.juju
    sudo docker run -v ~/.juju:/home/ubuntu/.juju -ti whitmo/jujubox:latest

At this point from either path you will have access to the `juju
quickstart` command.

To set up the credentials for your chosen cloud run:

    juju quickstart --constraints="mem=3.75G" -i

Follow the dialogue and choose `save` and `use`.  Quickstart will now
bootstrap the juju root node and setup the juju web based user
interface.


## Launch Kubernetes cluster

You will need to have the Kubernetes tools compiled before launching the cluster

    make all WHAT=cmd/kubectl
    export KUBERNETES_PROVIDER=juju
    cluster/kube-up.sh

If this is your first time running the `kube-up.sh` script, it will install
the required predependencies to get started with Juju, additionally it will
launch a curses based configuration utility allowing you to select your cloud
provider and enter the proper access credentials.

Next it will deploy the kubernetes master, etcd, 2 minions with flannel based
Software Defined Networking.


## Exploring the cluster

Juju status provides information about each unit in the cluster:

    juju status --format=oneline
    - docker/0: 52.4.92.78 (started)
      - flannel-docker/0: 52.4.92.78 (started)
      - kubernetes/0: 52.4.92.78 (started)
    - docker/1: 52.6.104.142 (started)
      - flannel-docker/1: 52.6.104.142 (started)
      - kubernetes/1: 52.6.104.142 (started)
    - etcd/0: 52.5.216.210 (started) 4001/tcp
    - juju-gui/0: 52.5.205.174 (started) 80/tcp, 443/tcp
    - kubernetes-master/0: 52.6.19.238 (started) 8080/tcp

You can use `juju ssh` to access any of the units:

    juju ssh kubernetes-master/0


## Run some containers!

`kubectl` is available on the kubernetes master node.  We'll ssh in to
launch some containers, but one could use kubectl locally setting
KUBERNETES_MASTER to point at the ip of `kubernetes-master/0`.

No pods will be available before starting a container:

    kubectl get pods
    POD  CONTAINER(S)   IMAGE(S)   HOST  LABELS  STATUS

    kubectl get replicationControllers
    CONTROLLER  CONTAINER(S)  IMAGE(S)  SELECTOR  REPLICAS

We'll follow the aws-coreos example. Create a pod manifest: `pod.json`

```
{
  "id": "hello",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "hello",
      "containers": [{
        "name": "hello",
        "image": "quay.io/kelseyhightower/hello",
        "ports": [{
          "containerPort": 80,
          "hostPort": 80
        }]
      }]
    }
  },
  "labels": {
    "name": "hello",
    "environment": "testing"
  }
}
```

Create the pod with kubectl:

    kubectl create -f pod.json


Get info on the pod:

    kubectl get pods


To test the hello app, we'll need to locate which minion is hosting
the container. Better tooling for using juju to introspect container
is in the works but for let'suse `juju run` and `juju status` to find
our hello app.

Exit out of our ssh session and run:

    juju run --unit kubernetes/0 "docker ps -n=1"
    ...
    juju run --unit kubernetes/1 "docker ps -n=1"
    CONTAINER ID        IMAGE                                  COMMAND             CREATED             STATUS              PORTS               NAMES
    02beb61339d8        quay.io/kelseyhightower/hello:latest   /hello              About an hour ago   Up About an hour                        k8s_hello....


We see `kubernetes/1` has our container, we can open port 80:

    juju run --unit kubernetes/1 "open-port 80"
    juju expose kubernetes
    sudo apt-get install curl
    curl $(juju status --format=oneline kubernetes/1 | cut -d' ' -f3)

Finally delete the pod:

    juju ssh kubernetes-master/0
    kubectl delete pods hello


## Scale out cluster

We can add minion units like so:

    juju add-unit docker # creates unit docker/2, kubernetes/2, docker-flannel/2


## Tear down cluster

    juju destroy-environment --force `juju env`


## More Info

Kubernetes Bundle on Github

 - [Bundle Repository](https://github.com/whitmo/bundle-kubernetes)
   * [Kubernetes master charm](https://github.com/whitmo/charm-kubernetes-master)
   * [Kubernetes mininion charm](https://github.com/whitmo/charm-kubernetes)
 - [Bundle Documentation](http://whitmo.github.io/bundle-kubernetes)
 - [More about Juju](https://juju.ubuntu.com)


### Cloud compatibility

Juju runs natively against a variety of cloud providers and can be
made to work against many more using a generic manual provider.

Provider          | v0.15.0
--------------    | -------
AWS               | TBD
HPCloud           | TBD
OpenStack         | TBD
Joyent            | TBD
Azure             | TBD
Digital Ocean     | TBD
MAAS (bare metal) | TBD
GCE               | TBD


Provider          | v0.8.1
--------------    | -------
AWS               | [Pass](http://reports.vapour.ws/charm-test-details/charm-bundle-test-parent-136)
HPCloud           | [Pass](http://reports.vapour.ws/charm-test-details/charm-bundle-test-parent-136)
OpenStack         | [Pass](http://reports.vapour.ws/charm-test-details/charm-bundle-test-parent-136)
Joyent            | [Pass](http://reports.vapour.ws/charm-test-details/charm-bundle-test-parent-136)
Azure             | TBD
Digital Ocean     | TBD
MAAS (bare metal) | TBD
GCE               | TBD
