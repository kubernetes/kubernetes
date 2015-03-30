## LMKTFY Namespaces

LMKTFY Namespaces help different projects, teams, or customers to share a LMKTFY cluster.

It does this by providing the following:

1. A scope for [Names](../../docs/identifiers.md).
2. A mechanism to attach authorization and policy to a subsection of the cluster.

Use of multiple namespaces is optional.

This example demonstrates how to use LMKTFY namespaces to subdivide your cluster.

### Step Zero: Prerequisites

This example assumes the following:

1. You have an existing LMKTFY cluster.
2. You have a basic understanding of LMKTFY pods, services, and replication controllers.

### Step One: Understand the default namespace

By default, a LMKTFY cluster will instantiate a default namespace when provisioning the cluster to hold the default set of pods,
services, and replication controllers used by the cluster.

Assuming you have a fresh cluster, you can introspect the available namespace's by doing the following:

```shell
$ cluster/lmktfyctl.sh get namespaces
NAME                LABELS
default             <none>
```

### Step Two: Create new namespaces

For this exercise, we will create two additional LMKTFY namespaces to hold our content.

Let's imagine a scenario where an organization is using a shared LMKTFY cluster for development and production use cases.

The development team would like to maintain a space in the cluster where they can get a view on the list of pods, services, and replication-controllers
they use to build and run their application.  In this space, LMKTFY resources come and go, and the restrictions on who can or cannot modify resources
are relaxed to enable agile development.

The operations team would like to maintain a space in the cluster where they can enforce strict procedures on who can or cannot manipulate the set of
pods, services, and replication controllers that run the production site.

One pattern this organization could follow is to partition the LMKTFY cluster into two namespaces: development and production.

Let's create two new namespaces to hold our work.

Use the file `examples/lmktfy-namespaces/namespace-dev.json` which describes a development namespace:

```js
{
  "kind": "Namespace",
  "apiVersion":"v1beta1",
  "id": "development",
  "spec": {},
  "status": {},
  "labels": {
    "name": "development"
  },
}
```

Create the development namespace using lmktfyctl.

```shell
$ cluster/lmktfyctl.sh create -f examples/lmktfy-namespaces/namespace-dev.json
```

And then lets create the production namespace using lmktfyctl.

```shell
$ cluster/lmktfyctl.sh create -f examples/lmktfy-namespaces/namespace-prod.json
```

To be sure things are right, let's list all of the namespaces in our cluster.

```shell
$ cluster/lmktfyctl.sh get namespaces
NAME                LABELS
default             <none>
development         name=development
production          name=production
```

### Step Three: Create pods in each namespace

A LMKTFY namespace provides the scope for pods, services, and replication controllers in the cluster.

Users interacting with one namespace do not see the content in another namespace.

To demonstrate this, let's spin up a simple replication controller and pod in the development namespace.

The first step is to define a context for the lmktfyctl client to work in each namespace.

```shell
$ cluster/lmktfyctl.sh config set-context dev --namespace=development
$ cluster/lmktfyctl.sh config set-context prod --namespace=production
```

The above commands provided two request contexts you can alternate against depending on what namespace you
wish to work against.

Let's switch to operate in the development namespace.

```shell
$ cluster/lmktfyctl.sh config use-context dev
```

You can verify your current context by doing the following:

```shell
$ cluster/lmktfyctl.sh config view
clusters: {}
contexts:
  dev:
    cluster: ""
    namespace: development
    user: ""
  prod:
    cluster: ""
    namespace: production
    user: ""
current-context: dev
preferences: {}
users: {}
```

At this point, all requests we make to the LMKTFY cluster from the command line are scoped to the development namespace.

Let's create some content.

```shell
$ cluster/lmktfyctl.sh run-container snowflake --image=lmktfy/serve_hostname --replicas=2
```

We have just created a replication controller whose replica size is 2 that is running the pod called snowflake with a basic container that just serves the hostname.

```shell
cluster/lmktfyctl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)                    SELECTOR                  REPLICAS
snowflake           snowflake           lmktfy/serve_hostname   run-container=snowflake   2

$ cluster/lmktfyctl.sh get pods
POD                 IP                  CONTAINER(S)        IMAGE(S)                    HOST                    LABELS                    STATUS
snowflake-fplln     10.246.0.5          snowflake           lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=snowflake   Running
snowflake-gziey     10.246.0.4          snowflake           lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=snowflake   Running
```

And this is great, developers are able to do what they want, and they do not have to worry about affecting content in the production namespace.

Let's switch to the production namespace and show how resources in one namespace are hidden from the other.

```shell
$ cluster/lmktfyctl.sh config use-context prod
```

The production namespace should be empty.

```shell
$ cluster/lmktfyctl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)            SELECTOR            REPLICAS

$ cluster/lmktfyctl.sh get pods
POD                 IP                  CONTAINER(S)        IMAGE(S)            HOST                LABELS              STATUS
```

Production likes to run cattle, so let's create some cattle pods.

```shell
$ cluster/lmktfyctl.sh run-container cattle --image=lmktfy/serve_hostname --replicas=5

$ cluster/lmktfyctl.sh get rc
CONTROLLER          CONTAINER(S)        IMAGE(S)                    SELECTOR               REPLICAS
cattle              cattle              lmktfy/serve_hostname   run-container=cattle   5

$ cluster/lmktfyctl.sh get pods
POD                 IP                  CONTAINER(S)        IMAGE(S)                    HOST                    LABELS                 STATUS
cattle-0133o        10.246.0.7          cattle              lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=cattle   Running
cattle-hh2gd        10.246.0.10         cattle              lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=cattle   Running
cattle-ls6k1        10.246.0.9          cattle              lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=cattle   Running
cattle-nyxxv        10.246.0.8          cattle              lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=cattle   Running
cattle-oh43e        10.246.0.6          cattle              lmktfy/serve_hostname   10.245.1.3/10.245.1.3   run-container=cattle   Running
```

At this point, it should be clear that the resources users create in one namespace are hidden from the other namespace.

As the policy support in LMKTFY evolves, we will extend this scenario to show how you can provide different
authorization rules for each namespace.
