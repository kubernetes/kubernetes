## Kubernetes Namespaces

Kubernetes Namespaces help different projects, teams, or customers to share a Kubernetes cluster.

It does this by providing the following:

1. A scope for [Names](../../docs/identifiers.md).
2. A mechanism to attach authorization and policy to a subsection of the cluster.

Use of multiple namespaces is optional.

This example demonstrates how to use Kubernetes namespaces to subdivide your cluster.

### Step Zero: Prerequisites

This example assumes the following:

1. You have an existing Kubernetes cluster.
2. You have a basic understanding of Kubernetes pods, services, and replication controllers.

### Step One: Understand the default namespace

By default, a Kubernetes cluster will instantiate a default namespace when provisioning the cluster to hold the default set of pods,
services, and replication controllers used by the cluster.

Assuming you have a fresh cluster, you can introspect the available namespace's by doing the following:

```shell
$ cluster/kubectl.sh get namespaces
NAME                LABELS
default             <none>
```

### Step Two: Create new namespaces

For this exercise, we will create two additional Kubernetes namespaces to hold our content.

Let's imagine a scenario where an organization is using a shared Kubernetes cluster for development and production use cases.

The development team would like to maintain a space in the cluster where they can get a view on the list of pods, services, and replication-controllers
they use to build and run their application.  In this space, Kubernetes resources come and go, and the restrictions on who can or cannot modify resources
are relaxed to enable agile development.

The operations team would like to maintain a space in the cluster where they can enforce strict procedures on who can or cannot manipulate the set of
pods, services, and replication controllers that run the production site.

One pattern this organization could follow is to partition the Kubernetes cluster into two namespaces: development and production.

Let's create two new namespaces to hold our work.

Use the file `examples/kubernetes-namespaces/namespace-dev.json` which describes a development namespace:

```js
{
  "kind": "Namespace",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "development",
    "labels": {
      "name": "development"
    }
  }
}
```

Create the development namespace using kubectl.

```shell
$ cluster/kubectl.sh create -f examples/kubernetes-namespaces/namespace-dev.json
```

And then lets create the production namespace using kubectl.

```shell
$ cluster/kubectl.sh create -f examples/kubernetes-namespaces/namespace-prod.json
```

To be sure things are right, let's list all of the namespaces in our cluster.

```shell
$ cluster/kubectl.sh get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
```


### Step Three: Create pods in each namespace

A Kubernetes namespace provides the scope for pods, services, and replication controllers in the cluster.

Users interacting with one namespace do not see the content in another namespace.

To demonstrate this, let's spin up a simple replication controller and pod in the development namespace.

We first check what is the current context:

```shell
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: REDACTED
    server: https://130.211.122.180
  name: lithe-cocoa-92103_kubernetes
contexts:
- context:
    cluster: lithe-cocoa-92103_kubernetes
    user: lithe-cocoa-92103_kubernetes
  name: lithe-cocoa-92103_kubernetes
current-context: lithe-cocoa-92103_kubernetes
kind: Config
preferences: {}
users:
- name: lithe-cocoa-92103_kubernetes
  user:
    client-certificate-data: REDACTED
    client-key-data: REDACTED
    token: 65rZW78y8HbwXXtSXuUw9DbP4FLjHi4b
- name: lithe-cocoa-92103_kubernetes-basic-auth
  user:
    password: h5M0FtUUIflBSdI7
    username: admin
```

The next step is to define a context for the kubectl client to work in each namespace. The value of "cluster" and "user" fields are copied from the current context.

```shell
$ cluster/kubectl.sh config set-context dev --namespace=development --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
$ cluster/kubectl.sh config set-context prod --namespace=production --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
```

The above commands provided two request contexts you can alternate against depending on what namespace you
wish to work against.

Let's switch to operate in the development namespace.

```shell
$ cluster/kubectl.sh config use-context dev
```

You can verify your current context by doing the following:

```shell
$ cluster/kubectl.sh config view
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: REDACTED
    server: https://130.211.122.180
  name: lithe-cocoa-92103_kubernetes
contexts:
- context:
    cluster: lithe-cocoa-92103_kubernetes
    namespace: development
    user: lithe-cocoa-92103_kubernetes
  name: dev
- context:
    cluster: lithe-cocoa-92103_kubernetes
    user: lithe-cocoa-92103_kubernetes
  name: lithe-cocoa-92103_kubernetes
- context:
    cluster: lithe-cocoa-92103_kubernetes
    namespace: production
    user: lithe-cocoa-92103_kubernetes
  name: prod
current-context: dev
kind: Config
preferences: {}
users:
- name: lithe-cocoa-92103_kubernetes
  user:
    client-certificate-data: REDACTED
    client-key-data: REDACTED
    token: 65rZW78y8HbwXXtSXuUw9DbP4FLjHi4b
- name: lithe-cocoa-92103_kubernetes-basic-auth
  user:
    password: h5M0FtUUIflBSdI7
    username: admin
```

At this point, all requests we make to the Kubernetes cluster from the command line are scoped to the development namespace.

Let's create some content.

```shell
$ cluster/kubectl.sh run-container snowflake --image=kubernetes/serve_hostname --replicas=2
```

We have just created a replication controller whose replica size is 2 that is running the pod called snowflake with a basic container that just serves the hostname.

```shell
cluster/kubectl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)                    SELECTOR                  REPLICAS
snowflake           snowflake           kubernetes/serve_hostname   run-container=snowflake   2

$ cluster/kubectl.sh get pods
POD               IP           CONTAINER(S)   IMAGE(S)                    HOST                                   LABELS                    STATUS    CREATED         MESSAGE
snowflake-mbrfi   10.244.2.4                                              kubernetes-minion-ilqx/104.197.8.214   run-container=snowflake   Running   About an hour   
                               snowflake      kubernetes/serve_hostname                                                                    Running   About an hour   
snowflake-p78ev   10.244.2.5                                              kubernetes-minion-ilqx/104.197.8.214   run-container=snowflake   Running   About an hour   
                               snowflake      kubernetes/serve_hostname                                                                    Running   About an hour   
```

And this is great, developers are able to do what they want, and they do not have to worry about affecting content in the production namespace.

Let's switch to the production namespace and show how resources in one namespace are hidden from the other.

```shell
$ cluster/kubectl.sh config use-context prod
```

The production namespace should be empty.

```shell
$ cluster/kubectl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)            SELECTOR            REPLICAS

$ cluster/kubectl.sh get pods
POD                 IP                  CONTAINER(S)        IMAGE(S)            HOST                LABELS              STATUS          CREATED       MESSAGE
```

Production likes to run cattle, so let's create some cattle pods.

```shell
$ cluster/kubectl.sh run-container cattle --image=kubernetes/serve_hostname --replicas=5

$ cluster/kubectl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)                    SELECTOR               REPLICAS
cattle              cattle              kubernetes/serve_hostname   run-container=cattle   5

$ cluster/kubectl.sh get pods
POD            IP           CONTAINER(S)   IMAGE(S)                    HOST                                    LABELS                 STATUS    CREATED         MESSAGE
cattle-1kyvj   10.244.0.4                                              kubernetes-minion-7s1y/23.236.54.97     run-container=cattle   Running   About an hour   
                            cattle         kubernetes/serve_hostname                                                                  Running   About an hour   
cattle-kobrk   10.244.1.4                                              kubernetes-minion-cfs6/104.154.61.231   run-container=cattle   Running   About an hour   
                            cattle         kubernetes/serve_hostname                                                                  Running   About an hour   
cattle-l1v9t   10.244.0.5                                              kubernetes-minion-7s1y/23.236.54.97     run-container=cattle   Running   About an hour   
                            cattle         kubernetes/serve_hostname                                                                  Running   About an hour   
cattle-ne2sj   10.244.3.7                                              kubernetes-minion-x8gx/104.154.47.83    run-container=cattle   Running   About an hour   
                            cattle         kubernetes/serve_hostname                                                                  Running   About an hour   
cattle-qrk4x   10.244.0.6                                              kubernetes-minion-7s1y/23.236.54.97     run-container=cattle   Running   About an hour   
                            cattle         kubernetes/serve_hostname  
```

At this point, it should be clear that the resources users create in one namespace are hidden from the other namespace.

As the policy support in Kubernetes evolves, we will extend this scenario to show how you can provide different
authorization rules for each namespace.
