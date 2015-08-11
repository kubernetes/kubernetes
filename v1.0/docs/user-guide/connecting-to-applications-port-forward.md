---
layout: docwithnav
title: "Connecting to applications: kubectl port-forward"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Connecting to applications: kubectl port-forward

kubectl port-forward forwards connections to a local port to a port on a pod. Its man page is available [here](kubectl/kubectl_port-forward.html). Compared to [kubectl proxy](accessing-the-cluster.html#using-kubectl-proxy), `kubectl port-forward` is more generic as it can forward TCP traffic while `kubectl proxy` can only forward HTTP traffic. This guide demonstrates how to use `kubectl port-forward` to connect to a Redis database, which may be useful for database debugging.

## Creating a Redis master

{% highlight console %}
{% raw %}
$ kubectl create examples/redis/redis-master.yaml
pods/redis-master
{% endraw %}
{% endhighlight %}

wait until the Redis master pod is Running and Ready,

{% highlight console %}
{% raw %}
$ kubectl get pods
NAME           READY     STATUS    RESTARTS   AGE
redis-master   2/2       Running   0          41s
{% endraw %}
{% endhighlight %}


## Connecting to the Redis master[a]

The Redis master is listening on port 6397, to verify this,

{% highlight console %}
{% raw %}
$ kubectl get pods redis-master -t='{{(index (index .spec.containers 0).ports 0).containerPort}}{{"\n"}}'
6379
{% endraw %}
{% endhighlight %}


then we forward the port 6379 on the local workstation to the port 6379 of pod redis-master,

{% highlight console %}
{% raw %}
$ kubectl port-forward -p redis-master 6379:6379
I0710 14:43:38.274550    3655 portforward.go:225] Forwarding from 127.0.0.1:6379 -> 6379
I0710 14:43:38.274797    3655 portforward.go:225] Forwarding from [::1]:6379 -> 6379
{% endraw %}
{% endhighlight %}

To verify the connection is successful, we run a redis-cli on the local workstation,

{% highlight console %}
{% raw %}
$ redis-cli
127.0.0.1:6379> ping
PONG
{% endraw %}
{% endhighlight %}

Now one can debug the database from the local workstation.


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/connecting-to-applications-port-forward.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

