---
layout: docwithnav
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Testing your Kubernetes cluster.

To validate that your node(s) have been added, run:

{% highlight sh %}
kubectl get nodes
{% endhighlight %}

That should show something like:

{% highlight console %}
NAME           LABELS                                 STATUS
10.240.99.26   kubernetes.io/hostname=10.240.99.26    Ready
127.0.0.1      kubernetes.io/hostname=127.0.0.1       Ready
{% endhighlight %}

If the status of any node is `Unknown` or `NotReady` your cluster is broken, double check that all containers are running properly, and if all else fails, contact us on IRC at
[`#google-containers`](http://webchat.freenode.net/?channels=google-containers) for advice.

### Run an application

{% highlight sh %}
kubectl -s http://localhost:8080 run nginx --image=nginx --port=80
{% endhighlight %}

now run `docker ps` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service

{% highlight sh %}
kubectl expose rc nginx --port=80
{% endhighlight %}

This should print:

{% highlight console %}
NAME      LABELS    SELECTOR              IP          PORT(S)
nginx     <none>    run=nginx             <ip-addr>   80/TCP
{% endhighlight %}

Hit the webserver:

{% highlight sh %}
curl <insert-ip-from-above-here>
{% endhighlight %}

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### Scaling 

Now try to scale up the nginx you created before:

{% highlight sh %}
kubectl scale rc nginx --replicas=3
{% endhighlight %}

And list the pods

{% highlight sh %}
kubectl get pods
{% endhighlight %}

You should see pods landing on the newly added machine.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/testing.html?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

