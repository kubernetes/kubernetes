---
layout: docwithnav
title: "Kubernetes Namespaces"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Kubernetes Namespaces

Kubernetes _[namespaces](../../../docs/admin/namespaces.html)_ help different projects, teams, or customers to share a Kubernetes cluster.

It does this by providing the following:

1. A scope for [Names](../../user-guide/identifiers.html).
2. A mechanism to attach authorization and policy to a subsection of the cluster.

Use of multiple namespaces is optional.

This example demonstrates how to use Kubernetes namespaces to subdivide your cluster.

### Step Zero: Prerequisites

This example assumes the following:

1. You have an [existing Kubernetes cluster](../../getting-started-guides/).
2. You have a basic understanding of Kubernetes _[pods](../../user-guide/pods.html)_, _[services](../../user-guide/services.html)_, and _[replication controllers](../../user-guide/replication-controller.html)_.

### Step One: Understand the default namespace

By default, a Kubernetes cluster will instantiate a default namespace when provisioning the cluster to hold the default set of pods,
services, and replication controllers used by the cluster.

Assuming you have a fresh cluster, you can introspect the available namespace's by doing the following:

{% highlight console %}
{% raw %}
$ kubectl get namespaces
NAME                LABELS
default             <none>
{% endraw %}
{% endhighlight %}

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

Use the file [`namespace-dev.json`](namespace-dev.json) which describes a development namespace:

{% highlight json %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

Create the development namespace using kubectl.

{% highlight console %}
{% raw %}
$ kubectl create -f docs/admin/namespaces/namespace-dev.json
{% endraw %}
{% endhighlight %}

And then lets create the production namespace using kubectl.

{% highlight console %}
{% raw %}
$ kubectl create -f docs/admin/namespaces/namespace-prod.json
{% endraw %}
{% endhighlight %}

To be sure things are right, let's list all of the namespaces in our cluster.

{% highlight console %}
{% raw %}
$ kubectl get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
{% endraw %}
{% endhighlight %}


### Step Three: Create pods in each namespace

A Kubernetes namespace provides the scope for pods, services, and replication controllers in the cluster.

Users interacting with one namespace do not see the content in another namespace.

To demonstrate this, let's spin up a simple replication controller and pod in the development namespace.

We first check what is the current context:

{% highlight yaml %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

The next step is to define a context for the kubectl client to work in each namespace. The value of "cluster" and "user" fields are copied from the current context.

{% highlight console %}
{% raw %}
$ kubectl config set-context dev --namespace=development --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
$ kubectl config set-context prod --namespace=production --cluster=lithe-cocoa-92103_kubernetes --user=lithe-cocoa-92103_kubernetes
{% endraw %}
{% endhighlight %}

The above commands provided two request contexts you can alternate against depending on what namespace you
wish to work against.

Let's switch to operate in the development namespace.

{% highlight console %}
{% raw %}
$ kubectl config use-context dev
{% endraw %}
{% endhighlight %}

You can verify your current context by doing the following:

{% highlight console %}
{% raw %}
$ kubectl config view
{% endraw %}
{% endhighlight %}

{% highlight yaml %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

At this point, all requests we make to the Kubernetes cluster from the command line are scoped to the development namespace.

Let's create some content.

{% highlight console %}
{% raw %}
$ kubectl run snowflake --image=kubernetes/serve_hostname --replicas=2
{% endraw %}
{% endhighlight %}

We have just created a replication controller whose replica size is 2 that is running the pod called snowflake with a basic container that just serves the hostname.

{% highlight console %}
{% raw %}
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR        REPLICAS
snowflake    snowflake      kubernetes/serve_hostname   run=snowflake   2

$ kubectl get pods
NAME              READY     STATUS    RESTARTS   AGE
snowflake-8w0qn   1/1       Running   0          22s
snowflake-jrpzb   1/1       Running   0          22s
{% endraw %}
{% endhighlight %}

And this is great, developers are able to do what they want, and they do not have to worry about affecting content in the production namespace.

Let's switch to the production namespace and show how resources in one namespace are hidden from the other.

{% highlight console %}
{% raw %}
$ kubectl config use-context prod
{% endraw %}
{% endhighlight %}

The production namespace should be empty.

{% highlight console %}
{% raw %}
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR   REPLICAS

$ kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
{% endraw %}
{% endhighlight %}

Production likes to run cattle, so let's create some cattle pods.

{% highlight console %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

At this point, it should be clear that the resources users create in one namespace are hidden from the other namespace.

As the policy support in Kubernetes evolves, we will extend this scenario to show how you can provide different
authorization rules for each namespace.


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/namespaces/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

