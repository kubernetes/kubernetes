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

If the status of any node is `Unknown` or `NotReady` your cluster is broken, double check that all containers are running properly, and if all else fails, contact us on [Slack](../../troubleshooting.html#slack).

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

Run the following command to obtain the IP of this service we just created. There are two IPs, the first one is internal (CLUSTER_IP), and the second one is the external load-balanced IP.

{% highlight sh %}
{% raw %}
kubectl get svc nginx
{% endraw %}
{% endhighlight %}

Alternatively, you can obtain only the first IP (CLUSTER_IP) by running:

{% highlight sh %}
{% raw %}
kubectl get svc nginx --template={{.spec.clusterIP}}
{% endraw %}
{% endhighlight %}

Hit the webserver with the first IP (CLUSTER_IP):

{% highlight sh %}
{% raw %}
curl <insert-cluster-ip-here>
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

