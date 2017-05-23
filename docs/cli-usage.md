## Using Kubernetes from the command line

A how to guide for accessing and running applications on a kubernetes cluster.

### Basic Setup
The kubernetes cluster is accessed via the ```kubectl`` command.  Full details of using the ```kubectl``` command are described [elsewhere](docs/cli.md).  Make sure that ```kubectl``` is in your path by running

```sh
which kubectl
```

If that command returns a path to the binary, you are ready to go.  If it doesn't, you need to update your PATH to include the appropriate directory from the Kubernetes directory.

### Commands
#### run
```sh
kubectl run <pod-name> --image=nginx [--replicas=N] [--labels=<key>=<value>,...][--export [--external]] [--namespace=<namespace>]
```

This translates into a call to create a replication controller and optionally, a call to create a service which exposes the pods created by that replication controller.  If ```--labels``` is unspecified, the Pods created by the replication controller simply have the label of ```name=<pod-name>```.  Otherwise, the user provided labels are used.  If ```--namespace``` is specified, the provided namespace is used.  Otherwise the currently configured namespace is used.  See ```kubectl config``` for details on how to set the current namespace.


#### describe
```sh
kubectl describe <name>
```

Describe attempts a get against the three resource types (Pods, Replication Controllers, Services) and returns matching resource(s) for the named resource.  It ignores 404 errors, unless all of the get requests return 404.


#### resize
```sh
kubectl resize <name> --count=N
```

Resize is used to either update the size of any resource that supports a resize operation.  Currently this is only Replica Controllers, though we anticipate adding additional resizable resources in the future.

#### export
```sh
kubectl export <service-name> <name> [--external] [--public-ip=<ip>] [--create-balancer=<bool>]
```

Export is used to create a service which represents a replicated set of pods.  First the replication controller is looked up using <name> then a new service called <service-name> is created.  If --external is specified, the service is also exposed on a public IP address, if --public-ip is specified, it is used as the ip address.  If --create-balancer is true (the default) an attempt to create a cloud load balancer with that IP address is performed.

#### label
```sh
kubectl label <name> <key-1>=<value-1> ... <key-n>=<value-n>
kubectl rmlabel <name> <key-1 ... <key-n>
```

Add, removes or updates labels to the api object named <name>. If more than one such object exists, returns an error, in which case the user must specify a complete path (e.g. ```pods/<name>```)


#### delete
```sh
kubectl delete <name>
```

Deletes an api object specified by ```<name>```.  Prompts the user for confirmation before deleting.  


### Usage Example
Consider setting up the [Guestbook example](examples/guestbook/README.md).

#### Create and export the master
```sh
kubectl run redis-master --image=dockerfile/redis --export
```

#### Create and export the slaves
```sh
kubectl run redis-slave --image=brendanburns/redis-slave --replicas=2 --export
```

#### Create and export the frontend
```sh
kubectl run frontend --image=php/redis --replicas=3 --export --external
```

### Resource Naming and Discovery

In many places we accept a ```<name>``` for a resource.  Sometimes the type of resource is implicit in the command (e.g. ```run``` or ```export```) but often it is ambiguous. When it is ambiguous (e.g. ```delete```) the kubectl tool does most of the work of determining which resource the user is requesting.  

Given this goal, the resource discovery algorithm works as follows:

If ```<name>`` has no slashes '/' in it, it is both without namespace and without a resource type.  In this case we first do a search in each resource type (pods, replica controllers, services)and the default namespace ('default') for an object with that name.  If this search fails, we return "not found" and ask the user if the object exists in a namespace.

If ```<name>``` has a single slash ('/') we split the name into prefix and suffix.  If prefix is one of
  * pod
  * replicacontroller
  * service
We assume that the prefix is a resource specification and search in that resource with the default namespace.

If the prefix is not one of the above, we assume that it is a namespace specification, and search all resource types with that namespace and name.

If ```<name>``` has two slashes, we split the name into ```<resource>/<namespace>/<name>```, and search for that item specifically, returning "not found" if it can't be found.
