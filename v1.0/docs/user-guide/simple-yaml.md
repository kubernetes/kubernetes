---
layout: docwithnav
title: "Getting started with config files."
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Getting started with config files.

In addition to the imperative style commands described [elsewhere](simple-nginx.html), Kubernetes
supports declarative YAML or JSON configuration files.  Often times config files are preferable
to imperative commands, since they can be checked into version control and changes to the files
can be code reviewed, producing a more robust, reliable and archival system.

### Running a container from a pod configuration file

{% highlight console %}
{% raw %}
$ cd kubernetes
$ kubectl create -f ./pod.yaml
{% endraw %}
{% endhighlight %}

Where pod.yaml contains something like:

{% highlight yaml %}
{% raw %}
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
{% endraw %}
{% endhighlight %}

You can see your cluster's pods:

{% highlight console %}
{% raw %}
$ kubectl get pods
{% endraw %}
{% endhighlight %}

and delete the pod you just created:

{% highlight console %}
{% raw %}
$ kubectl delete pods nginx
{% endraw %}
{% endhighlight %}

### Running a replicated set of containers from a configuration file

To run replicated containers, you need a [Replication Controller](replication-controller.html).
A replication controller is responsible for ensuring that a specific number of pods exist in the
cluster.

{% highlight console %}
{% raw %}
$ cd kubernetes
$ kubectl create -f ./replication.yaml
{% endraw %}
{% endhighlight %}

Where `replication.yaml` contains:

{% highlight yaml %}
{% raw %}
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
{% endraw %}
{% endhighlight %}

To delete the replication controller (and the pods it created):

{% highlight console %}
{% raw %}
$ kubectl delete rc nginx
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/simple-yaml.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

