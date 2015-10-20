---
layout: docwithnav
title: "Testing your Kubernetes cluster."
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Testing your Kubernetes cluster.

To validate that your node(s) have been added, run:

{% highlight sh %}
{% raw %}
kubectl get nodes
{% endraw %}
{% endhighlight %}

That should show something like:

{% highlight console %}
{% raw %}
NAME           LABELS                                 STATUS
10.240.99.26   kubernetes.io/hostname=10.240.99.26    Ready
127.0.0.1      kubernetes.io/hostname=127.0.0.1       Ready
{% endraw %}
{% endhighlight %}

If the status of any node is `Unknown` or `NotReady` your cluster is broken, double check that all containers are running properly, and if all else fails, contact us on IRC at
[`#google-containers`](http://webchat.freenode.net/?channels=google-containers) for advice.

### Run an application

{% highlight sh %}
{% raw %}
kubectl -s http://localhost:8080 run nginx --image=nginx --port=80
{% endraw %}
{% endhighlight %}

now run `docker ps` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service

{% highlight sh %}
{% raw %}
kubectl expose rc nginx --port=80
{% endraw %}
{% endhighlight %}

This should print:

{% highlight console %}
{% raw %}
NAME      LABELS    SELECTOR              IP          PORT(S)
nginx     <none>    run=nginx             <ip-addr>   80/TCP
{% endraw %}
{% endhighlight %}

Hit the webserver:

{% highlight sh %}
{% raw %}
curl <insert-ip-from-above-here>
{% endraw %}
{% endhighlight %}

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### Scaling 

Now try to scale up the nginx you created before:

{% highlight sh %}
{% raw %}
kubectl scale rc nginx --replicas=3
{% endraw %}
{% endhighlight %}

And list the pods

{% highlight sh %}
{% raw %}
kubectl get pods
{% endraw %}
{% endhighlight %}

You should see pods landing on the newly added machine.


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

