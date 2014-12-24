## Using Kubernetes from the command line

A how to guide for accessing and running applications on a kubernetes cluster.

### Basic Setup
The kubernetes cluster is accessed via the ```kubectl`` command.  Full details of using the ```kubectl``` command are described [elsewhere](docs/cli.md).  Make sure that ```kubectl``` is in your path by running

```sh
which kubectl
```

If that command returns a path to the binary, you are ready to go.  If it doesn't, you need to update your PATH to include the appropriate directory from the Kubernetes directory.

### Commands
#### Run
```sh
kubectl run <pod-name> --image=nginx [--replicas=N] [--export [--external]]
```

This translates into a call to create a replication controller and optionally, a call to create a service which exposes the pods created by that replication controller.


#### Describe
```sh
kubectl describe <name>
```

Describe attempts a get against the three resource types (Pods, Replication Controllers, Services) and returns matching resource(s) for the named resource.  It ignores 404 errors, unless all of the get requests return 404.


#### Update
```sh
kubectl update <name> [--replicas=N]
```

Update is used to either update the size of a replication controller.

#### Export
```sh
kubectl export <service-name> <name> [--external] [--public-ip=<ip>] [--create-balancer=<bool>]
```

Export is used to create a service which represents a replicated set of pods.  First the replication controller is looked up using <name> then a new service called <service-name> is created.  If --external is specified, the service is also exposed on a public IP address, if --public-ip is specified, it is used as the ip address.  If --create-balancer is true (the default) an attempt to create a cloud load balancer with that IP address is performed.

#### Label
```sh
kubectl label <name> <key-1>=<value-1> ... <key-n>=<value-n>
kubectl rmlabel <name> <key-1 ... <key-n>
```

Add, removes or updates labels to the api object named <name>. If more than one such object exists, returns an error, in which case the user must specify a complete path (e.g. ```pods/<name>```)


#### Delete
```sh
kubectl delete <name>
```

Deletes an api object specified by ```<name>```.  Prompts the user for confirmation before deleting.  


### Usage Example
