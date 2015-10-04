<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/kubectl-overview.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# kubectl overview

**Table of Contents**

- [kubectl overview](#kubectl-overview)
  - [Overview](#overview)
  - [Common Operations](#common-operations)
  - [Kubectl Operations](#kubectl-operations)
  - [Resource Types](#resource-types)

This overview is intended for anyone who wants to use `kubectl` command line tool to interact with Kubernetes cluster. Please remember that it is built for quick started with `kubectl`; for complete and detailed information, please refer to [kubectl](kubectl/kubectl.md).

TODO: auto-generate this file to stay up with `kubectl` changes. Please see [#14177](https://github.com/kubernetes/kubernetes/pull/14177).

## Overview

`kubectl` controls the Kubernetes cluster manager. The synopsis is:

```
kubectl [command] [TYPE] [NAME] [flags]
```

This specifies:

- `command` is a certain operation performed on a given resource(s), such as `create`, `get`, `describe`, `delete` etc.
- `TYPE` is the type of resource(s). Both singular and plural forms are accepted. For example, `node(s)`, `namespace(s)`, `pod(s)`, `replicationcontroller(s)`, `service(s)` etc.
- `NAME` is the name of resource(s). `TYPE NAME` can be specified as `TYPE name1 name2` or `TYPE/name1 TYPE/name2`. `TYPE NAME` can also be specified by one or more file arguments: `-f file1 -f file2 ...`, [use YAML rather than JSON](config-best-practices.md) since YAML tends to be more user-friendly for config.
- `flags` are used to provide more control information when running a command. For example, you can use `-s` or `--server` to specify the address and port of the Kubernetes API server. Command line flags override their corresponding default values and environment variables. [Use short flags sparingly, only for the most frequently used options](../devel/kubectl-conventions.md).

Please use `kubectl help [command]` for detailed information about a command.

Please refer to [kubectl](kubectl/kubectl.md) for a complete list of available commands and flags.

## Common Operations

For explanation, here I gave some mostly often used `kubectl` command examples. Please replace sample names with actual values if you would like to try these commands.

1. `kubectl create` - Create a resource by filename or stdin

		// Create a service using the data in example-service.yaml.
		$ kubectl create -f example-service.yaml

		// Create a replication controller using the data in example-controller.yaml.
		$ kubectl create -f example-controller.yaml

		// Create objects whose definitions are in a directory. This looks for config objects in all .yaml, .yml, and .json files in <directory> and passes them to create.
		$ kubectl create -f <directory>

2. `kubectl get` - Display one or many resources

		// List all pods in ps output format.
		$ kubectl get pods

		// List all pods in ps output format with more information (such as node name).
		$ kubectl get pods -o wide

		// List a single replication controller with specified name in ps output format. You can use the alias 'rc' instead of 'replicationcontroller'.
		$ kubectl get replicationcontroller <rc-name>

		// List all replication controllers and services together in ps output format.
		$ kubectl get rc,services

3. `kubectl describe` - Show details of a specific resource or group of resources

		// Describe a node
		$ kubectl describe nodes <node-name>

		// Describe a pod
		$ kubectl describe pods/<pod-name>

		// Describe all pods managed by the replication controller <rc-name>
		// (rc-created pods get the name of the rc as a prefix in the pod the name).
		$ kubectl describe pods <rc-name>

4. `kubectl delete` - Delete resources by filenames, stdin, resources and names, or by resources and label selector

		// Delete a pod using the type and name specified in pod.yaml.
		$ kubectl delete -f pod.yaml

		// Delete pods and services with label name=<label-name>.
		$ kubectl delete pods,services -l name=<label-name>

		// Delete all pods
		$ kubectl delete pods --all

5. `kubectl exec` - Execute a command in a container

		// Get output from running 'date' from pod <pod-name>, using the first container by default.
		$ kubectl exec <pod-name> date

		// Get output from running 'date' in <container-name> from pod <pod-name>.
		$ kubectl exec <pod-name> -c <container-name> date

		// Get an interactive tty and run /bin/bash from pod <pod-name>, using the first container by default.
		$ kubectl exec -ti <pod-name> /bin/bash

6. `kubectl logs` - Print the logs for a container in a pod.

		// Returns snapshot of logs from pod <pod-name>.
		$ kubectl logs <pod-name>

		// Starts streaming of logs from pod <pod-name>, it is something like 'tail -f'.
		$ kubectl logs -f <pod-name>

## Kubectl Operations

The following table describes all `kubectl` operations and their general synopsis:

Operation       | Synopsis	|       Description
-------------------- | -------------------- | --------------------
annotate	| `kubectl annotate [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]` | Update the annotations on a resource
api-versions	| `kubectl api-versions` | Print available API versions
attach		| `kubectl attach POD -c CONTAINER` | Attach to a running container
cluster-info	| `kubectl cluster-info` | Display cluster info
config		| `kubectl config SUBCOMMAND` | Modifies kubeconfig files
create		| `kubectl create -f FILENAME` | Create a resource by filename or stdin
delete		| `kubectl delete ([-f FILENAME] | TYPE [(NAME | -l label | --all)])` | Delete resources by filenames, stdin, resources and names, or by resources and label selector
describe	| `kubectl describe (-f FILENAME | TYPE [NAME_PREFIX | -l label] | TYPE/NAME)` | Show details of a specific resource or group of resources
edit		| `kubectl edit (RESOURCE/NAME | -f FILENAME)` | Edit a resource on the server
exec		| `kubectl exec POD [-c CONTAINER] -- COMMAND [args...]` | Execute a command in a container
expose		| `kubectl expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP] [--target-port=number-or-name] [--name=name] [----external-ip=external-ip-of-service] [--type=type]` | Take a replication controller, service or pod and expose it as a new Kubernetes Service
get		| `kubectl get [(-o|--output=)json|yaml|wide|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=...] (TYPE [NAME | -l label] | TYPE/NAME ...) [flags]` | Display one or many resources
label		| `kubectl label [--overwrite] (-f FILENAME | TYPE NAME) KEY_1=VAL_1 ... KEY_N=VAL_N [--resource-version=version]` | Update the labels on a resource
logs		| `kubectl logs [-f] [-p] POD [-c CONTAINER]` | Print the logs for a container in a pod
namespace	| `kubectl namespace [namespace]` | SUPERSEDED: Set and view the current Kubernetes namespace
patch		| `kubectl patch (-f FILENAME | TYPE NAME) -p PATCH` | Update field(s) of a resource by stdin
port-forward	| `kubectl port-forward POD [LOCAL_PORT:]REMOTE_PORT [...[LOCAL_PORT_N:]REMOTE_PORT_N]` | Forward one or more local ports to a pod
proxy		| `kubectl proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]` | Run a proxy to the Kubernetes API server
replace		| `kubectl replace -f FILENAME` | Replace a resource by filename or stdin
rolling-update	| `kubectl rolling-update OLD_CONTROLLER_NAME ([NEW_CONTROLLER_NAME] --image=NEW_CONTAINER_IMAGE | -f NEW_CONTROLLER_SPEC)` | Perform a rolling update of the given ReplicationController
run		| `kubectl run NAME --image=image [--env="key=value"] [--port=port] [--replicas=replicas] [--dry-run=bool] [--overrides=inline-json]` | Run a particular image on the cluster
scale		| `kubectl scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT (-f FILENAME | TYPE NAME)` | Set a new size for a Replication Controller
stop		| `kubectl stop (-f FILENAME | TYPE (NAME | -l label | --all))` | Deprecated: Gracefully shut down a resource by name or filename
version		| `kubectl version` | Print the client and server version information

## Resource Types

The `kubectl` supports the following resource types, and their abbreviated aliases:

Resource Type	| Abbreviated Alias
-------------------- | --------------------
componentstatuses	|	cs
events	|	ev
endpoints	|	ep
horizontalpodautoscalers	|	hpa
limitranges	|	limits
nodes	|	no
namespaces	|	ns
pods	|	po
persistentvolumes	|	pv
persistentvolumeclaims	|	pvc
resourcequotas	|	quota
replicationcontrollers	|	rc
daemonsets	|	ds
services	|	svc
ingress		|	ing

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl-overview.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
