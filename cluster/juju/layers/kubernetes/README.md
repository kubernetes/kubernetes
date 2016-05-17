# kubernetes

[Kubernetes](https://github.com/kubernetes/kubernetes) is an open
source system for managing application containers across multiple hosts.
This version of Kubernetes uses [Docker](http://www.docker.io/) to package,
instantiate and run containerized applications.

This charm is an encapsulation of the
[Running Kubernetes locally via
Docker](http://kubernetes.io/docs/getting-started-guides/docker)
document. The released hyperkube image (`gcr.io/google_containers/hyperkube`)
is currently pulled from a [Google owned container repository
repository](https://cloud.google.com/container-registry/).  For this charm to
work it will need access to the repository to `docker pull` the images.

This charm was built from other charm layers using the reactive framework. The
`layer:docker` is the base layer. For more information please read [Getting
Started Developing charms](https://jujucharms.com/docs/devel/developer-getting-started)

# Deployment
The kubernetes charms require a relation to a distributed key value store
(ETCD) which Kubernetes uses for persistent storage of all of its REST API
objects.

```
juju deploy trusty/etcd
juju deploy local:trusty/kubernetes
juju add-relation kubernetes etcd
```

# Configuration
For your convenience this charm supports some configuration options to set up
a Kuberentes cluster that works in your environment:  

**version**: Set the version of the Kubernetes containers to deploy.
The default value is "v1.0.6".  Changing the version causes the all the
Kubernetes containers to be restarted.

**cidr**: Set the IP range for the Kubernetes cluster. eg: 10.1.0.0/16


## State Events
While this charm is meant to be a top layer, it can be used to build other
solutions.  This charm sets or removes states from the reactive framework that
other layers could react appropriately. The states that other layers would be
interested in are as follows:

**kubelet.available** - The hyperkube container has been run with the kubelet
service and configuration that started the apiserver, controller-manager and
scheduler containers.

**proxy.available** - The hyperkube container has been run with the proxy
service and configuration that handles Kubernetes networking.

**kubectl.package.created** - Indicates the availability of the `kubectl`
application along with the configuration needed to contact the cluster
securely. You will need to download the `/home/ubuntu/kubectl_package.tar.gz`
from the kubernetes leader unit to your machine so you can control the cluster.

**skydns.available** - Indicates when the Domain Name System (DNS) for the
cluster is operational.


# Kubernetes information

 - [Kubernetes github project](https://github.com/kubernetes/kubernetes)
 - [Kubernetes issue tracker](https://github.com/kubernetes/kubernetes/issues)
 - [Kubernetes Documenation](http://kubernetes.io/docs/)
 - [Kubernetes releases](https://github.com/kubernetes/kubernetes/releases)

# Contact

 * Charm Author: Matthew Bruzek &lt;Matthew.Bruzek@canonical.com&gt;
 * Charm Contributor: Charles Butler &lt;Charles.Butler@canonical.com&gt;


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/juju/layers/kubernetes/README.md?pixel)]()
