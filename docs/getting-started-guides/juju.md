## Getting start with Juju

Juju handles provisioning machines and deploying complex systems to a
wide number of clouds.



### Prerequisites

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

    juju quickstart https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml

First this command will start a curses based gui allowing you to set
up credentials and other environmental settings for several different
providers including Azure and AWS.

Next it will deploy the kubernetes master, etcd, 2 minions with flannel networking.


## Exploring the cluster

Juju status provides information about each unit in the cluster:

    juju status --format=oneline

    - etcd/0: 52.0.74.109 (started)
    - flannel/0: 52.0.149.150 (started)
    - flannel/1: 52.0.185.81 (started)
    - juju-gui/0: 52.1.150.81 (started)
    - kubernetes/0: 52.0.149.150 (started)
    - kubernetes/1: 52.0.185.81 (started)
    - kubernetes-master/0: 52.1.120.142 (started)

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

    juju add-unit flannel # creates unit flannel/2
    juju add-unit kubernetes --to flannel/2


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

