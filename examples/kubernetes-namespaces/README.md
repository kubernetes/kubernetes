## Kubernetes Namespaces

Kubernetes _[namespaces](../../docs/namespaces.md)_ help different projects, teams, or customers to share a Kubernetes cluster.

It does this by providing the following:

1. A scope for [Names](../../docs/identifiers.md).
2. A mechanism to attach authorization and policy to a subsection of the cluster.

Use of multiple namespaces is optional.

This example demonstrates how to use Kubernetes namespaces to subdivide your cluster.

### Step Zero: Prerequisites

This example assumes the following:

1. You have an [existing Kubernetes cluster](../../docs/getting-started-guides).
2. You have a basic understanding of Kubernetes _[pods](../../docs/pods.md)_, _[services](../../docs/services.md)_, and _[replication controllers](../../docs/replication-controller.md)_.

### Step One: Understand the default namespace

By default, a Kubernetes cluster will instantiate a default namespace when provisioning the cluster to hold the default set of pods,
services, and replication controllers used by the cluster.

Assuming you have a fresh cluster, you can introspect the available namespace's by doing the following:

```shell
$ kubectl get namespaces
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

Use the file [`examples/kubernetes-namespaces/namespace-dev.json`](namespace-dev.json) which describes a development namespace:

```js
{
  "kind": "Namespace",
  "apiVersion": "v1",
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
$ kubectl create -f examples/kubernetes-namespaces/namespace-dev.json
```

And then lets create the production namespace using kubectl.

```shell
$ kubectl create -f examples/kubernetes-namespaces/namespace-prod.json
```

To be sure things are right, let's list all of the namespaces in our cluster.

```shell
$ kubectl get namespaces
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
$ kubectl config set-context dev --namespace=development --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
$ kubectl config set-context prod --namespace=production --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
```

The above commands provided two request contexts you can alternate against depending on what namespace you
wish to work against.

Let's switch to operate in the development namespace.

```shell
$ kubectl config use-context dev
```

You can verify your current context by doing the following:

```shell
$ kubectl config view
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
$ kubectl run snowflake --image=kubernetes/serve_hostname --replicas=2
```

We have just created a replication controller whose replica size is 2 that is running the pod called snowflake with a basic container that just serves the hostname.

```shell
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR        REPLICAS
snowflake    snowflake      kubernetes/serve_hostname   run=snowflake   2

$ kubectl get pods
NAME              READY     STATUS    RESTARTS   AGE
snowflake-8w0qn   1/1       Running   0          22s
snowflake-jrpzb   1/1       Running   0          22s
```

And this is great, developers are able to do what they want, and they do not have to worry about affecting content in the production namespace.

Let's switch to the production namespace and show how resources in one namespace are hidden from the other.

```shell
$ kubectl config use-context prod
```

The production namespace should be empty.

```shell
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR   REPLICAS

$ kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
```

Production likes to run cattle, so let's create some cattle pods.

```shell
$ kubectl run cattle --image=kubernetes/serve_hostname --replicas=5

$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR     REPLICAS
cattle       cattle         kubernetes/serve_hostname   run=cattle   5

$ kubectl get pods
NAME           READY     STATUS    RESTARTS   AGE
cattle-97rva   1/1       Running   0          12s
cattle-i9ojn   1/1       Running   0          12s
cattle-qj3yv   1/1       Running   0          12s
cattle-yc7vn   1/1       Running   0          12s
cattle-zz7ea   1/1       Running   0          12s
```

At this point, it should be clear that the resources users create in one namespace are hidden from the other namespace.

As the policy support in Kubernetes evolves, we will extend this scenario to show how you can provide different
authorization rules for each namespace.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/kubernetes-namespaces/README.md?pixel)]()
