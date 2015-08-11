---
layout: docwithnav
title: "Kubernetes DNS example"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Kubernetes DNS example

This is a toy example demonstrating how to use kubernetes DNS.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides/). Make sure DNS is enabled in your setup, see [DNS doc](../../cluster/addons/dns/).

{% highlight sh %}
{% raw %}
$ cd kubernetes
$ hack/dev-build-and-up.sh
{% endraw %}
{% endhighlight %}

### Step One: Create two namespaces

We'll see how cluster DNS works across multiple [namespaces](../../docs/user-guide/namespaces.html), first we need to create two namespaces:

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/cluster-dns/namespace-dev.yaml
$ kubectl create -f examples/cluster-dns/namespace-prod.yaml
{% endraw %}
{% endhighlight %}

Now list all namespaces:

{% highlight sh %}
{% raw %}
$ kubectl get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
{% endraw %}
{% endhighlight %}

For kubectl client to work with each namespace, we define two contexts:

{% highlight sh %}
{% raw %}
$ kubectl config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
$ kubectl config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}
{% endraw %}
{% endhighlight %}

You can view your cluster name and user name in kubernetes config at ~/.kube/config.

### Step Two: Create backend replication controller in each namespace

Use the file [`examples/cluster-dns/dns-backend-rc.yaml`](dns-backend-rc.yaml) to create a backend server [replication controller](../../docs/user-guide/replication-controller.html) in each namespace.

{% highlight sh %}
{% raw %}
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
{% endraw %}
{% endhighlight %}

Once that's up you can list the pod in the cluster:

{% highlight sh %}
{% raw %}
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
{% endraw %}
{% endhighlight %}

Now repeat the above commands to create a replication controller in prod namespace:

{% highlight sh %}
{% raw %}
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
{% endraw %}
{% endhighlight %}

### Step Three: Create backend service

Use the file [`examples/cluster-dns/dns-backend-service.yaml`](dns-backend-service.yaml) to create
a [service](../../docs/user-guide/services.html) for the backend server.

{% highlight sh %}
{% raw %}
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
{% endraw %}
{% endhighlight %}

Once that's up you can list the service in the cluster:

{% highlight sh %}
{% raw %}
$ kubectl get service dns-backend
NAME          LABELS    SELECTOR           IP(S)          PORT(S)
dns-backend   <none>    name=dns-backend   10.0.236.129   8000/TCP
{% endraw %}
{% endhighlight %}

Again, repeat the same process for prod namespace:

{% highlight sh %}
{% raw %}
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
$ kubectl get service dns-backend
NAME          LABELS    SELECTOR           IP(S)         PORT(S)
dns-backend   <none>    name=dns-backend   10.0.35.246   8000/TCP
{% endraw %}
{% endhighlight %}

### Step Four: Create client pod in one namespace

Use the file [`examples/cluster-dns/dns-frontend-pod.yaml`](dns-frontend-pod.yaml) to create a client [pod](../../docs/user-guide/pods.html) in dev namespace. The client pod will make a connection to backend and exit. Specifically, it tries to connect to address `http://dns-backend.development.cluster.local:8000`.

{% highlight sh %}
{% raw %}
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
{% endraw %}
{% endhighlight %}

Once that's up you can list the pod in the cluster:

{% highlight sh %}
{% raw %}
$ kubectl get pods dns-frontend
NAME           READY     STATUS       RESTARTS   AGE
dns-frontend   0/1       ExitCode:0   0          1m
{% endraw %}
{% endhighlight %}

Wait until the pod succeeds, then we can see the output from the client pod:

{% highlight sh %}
{% raw %}
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
{% endraw %}
{% endhighlight %}

Please refer to the [source code](images/frontend/client.py) about the log. First line prints out the ip address associated with the service in dev namespace; remaining lines print out our request and server response.

If we switch to prod namespace with the same pod config, we'll see the same result, i.e. dns will resolve across namespace.

{% highlight sh %}
{% raw %}
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
{% endraw %}
{% endhighlight %}


#### Note about default namespace

If you prefer not using namespace, then all your services can be addressed using `default` namespace, e.g. `http://dns-backend.default.cluster.local:8000`, or shorthand version `http://dns-backend:8000`


### tl; dr;

For those of you who are impatient, here is the summary of the commands we ran in this tutorial. Remember to set first `$CLUSTER_NAME` and `$USER_NAME` to the values found in `~/.kube/config`.

{% highlight sh %}
{% raw %}
# create dev and prod namespaces
kubectl create -f examples/cluster-dns/namespace-dev.yaml
kubectl create -f examples/cluster-dns/namespace-prod.yaml

# create two contexts
kubectl config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
kubectl config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}

# create two backend replication controllers
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-backend-rc.yaml

# create backend services
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-backend-service.yaml
kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-backend-service.yaml

# create a pod in each namespace and get its output
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
kubectl logs dns-frontend

kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
kubectl logs dns-frontend
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cluster-dns/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

