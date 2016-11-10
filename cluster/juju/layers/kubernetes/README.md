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
juju deploy etcd
juju deploy kubernetes
juju add-relation kubernetes etcd
```

# Configuration
For your convenience this charm supports some configuration options to set up
a Kubernetes cluster that works in your environment:  

**version**: Set the version of the Kubernetes containers to deploy. The
version string must be in the following format "v#.#.#" where the numbers
match with the
[kubernetes release labels](https://github.com/kubernetes/kubernetes/releases)
of the [kubernetes github project](https://github.com/kubernetes/kubernetes).
Changing the version causes the all the Kubernetes containers to be restarted.

**cidr**: Set the IP range for the Kubernetes cluster. eg: 10.1.0.0/16

**dns_domain**: Set the DNS domain for the Kubernetes cluster.

# Storage
The kubernetes charm is built to handle multiple storage devices if the cloud
provider works with
[Juju storage](https://jujucharms.com/docs/devel/charms-storage).

The 16.04 (xenial) release introduced [ZFS](https://en.wikipedia.org/wiki/ZFS)
to Ubuntu. The xenial charm can use ZFS witha raidz pool. A raidz pool
distributes parity along with the data (similar to a raid5 pool) and can suffer
the loss of one drive while still retaining data. The raidz pool requires a
minimum of 3 disks, but will accept more if they are provided.

You can add storage to the kubernetes charm in increments of 3 or greater:

```
juju add-storage kubernetes/0 disk-pool=ebs,3,1G
```

**Note**: Due to a limitation of raidz you can not add individual disks to an
existing pool. Should you need to expand the storage of the raidz pool, the
additional add-storage commands must be the same number of disks as the original
command. At this point the charm will have two raidz pools added together, both
of which could handle the  loss of one disk each.

The storage code handles the addition of devices to the charm and when it
receives three disks creates a raidz pool that is mounted at the /srv/kubernetes
directory by default. If you need the storage in another location you must
change the `mount-point` value in layer.yaml before the charms is deployed.

To avoid data loss you must attach the storage before making the connection to
the etcd cluster.

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

**kubedns.available** - Indicates when the Domain Name System (DNS) for the
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
