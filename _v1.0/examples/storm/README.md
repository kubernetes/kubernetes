---
layout: docwithnav
title: "Storm example"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Storm example

Following this example, you will create a functional [Apache
Storm](http://storm.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup an [Apache ZooKeeper](http://zookeeper.apache.org/)
service, a Storm master service (a.k.a. Nimbus server), and a set of
Storm workers (a.k.a. supervisors).

For the impatient expert, jump straight to the [tl;dr](#tldr)
section.

### Sources

Source is freely available at:
* Docker image - https://github.com/mattf/docker-storm
* Docker Trusted Build - https://registry.hub.docker.com/search?q=mattf/storm

## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and
running, and that you have installed the ```kubectl``` command line
tool somewhere in your path. Please see the [getting
started](../../docs/getting-started-guides/) for installation
instructions for your platform.

## Step One: Start your ZooKeeper service

ZooKeeper is a distributed coordination [service](../../docs/user-guide/services.html) that Storm uses as a
bootstrap and for state storage.

Use the [`examples/storm/zookeeper.json`](zookeeper.json) file to create a [pod](../../docs/user-guide/pods.html) running
the ZooKeeper service.

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/storm/zookeeper.json
{% endraw %}
{% endhighlight %}

Then, use the [`examples/storm/zookeeper-service.json`](zookeeper-service.json) file to create a
logical service endpoint that Storm can use to access the ZooKeeper
pod.

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/storm/zookeeper-service.json
{% endraw %}
{% endhighlight %}

You should make sure the ZooKeeper pod is Running and accessible
before proceeding.

### Check to see if ZooKeeper is running

{% highlight sh %}
{% raw %}
$ kubectl get pods
NAME        READY     STATUS    RESTARTS   AGE
zookeeper   1/1       Running   0          43s
{% endraw %}
{% endhighlight %}

### Check to see if ZooKeeper is accessible

{% highlight sh %}
{% raw %}
$ kubectl get services
NAME                LABELS                                    SELECTOR            IP(S)               PORT(S)
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
zookeeper           name=zookeeper                            name=zookeeper      10.254.139.141      2181

$ echo ruok | nc 10.254.139.141 2181; echo
imok
{% endraw %}
{% endhighlight %}

## Step Two: Start your Nimbus service

The Nimbus service is the master (or head) service for a Storm
cluster. It depends on a functional ZooKeeper service.

Use the [`examples/storm/storm-nimbus.json`](storm-nimbus.json) file to create a pod running
the Nimbus service.

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/storm/storm-nimbus.json
{% endraw %}
{% endhighlight %}

Then, use the [`examples/storm/storm-nimbus-service.json`](storm-nimbus-service.json) file to
create a logical service endpoint that Storm workers can use to access
the Nimbus pod.

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/storm/storm-nimbus-service.json
{% endraw %}
{% endhighlight %}

Ensure that the Nimbus service is running and functional.

### Check to see if Nimbus is running and accessible

{% highlight sh %}
{% raw %}
$ kubectl get services
NAME                LABELS                                    SELECTOR            IP(S)               PORT(S)
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
zookeeper           name=zookeeper                            name=zookeeper      10.254.139.141      2181
nimbus              name=nimbus                               name=nimbus         10.254.115.208      6627

$ sudo docker run -it -w /opt/apache-storm mattf/storm-base sh -c '/configure.sh 10.254.139.141 10.254.115.208; ./bin/storm list'
...
No topologies running.
{% endraw %}
{% endhighlight %}

## Step Three: Start your Storm workers

The Storm workers (or supervisors) do the heavy lifting in a Storm
cluster. They run your stream processing topologies and are managed by
the Nimbus service.

The Storm workers need both the ZooKeeper and Nimbus services to be
running.

Use the [`examples/storm/storm-worker-controller.json`](storm-worker-controller.json) file to create a
[replication controller](../../docs/user-guide/replication-controller.html) that manages the worker pods.

{% highlight sh %}
{% raw %}
$ kubectl create -f examples/storm/storm-worker-controller.json
{% endraw %}
{% endhighlight %}

### Check to see if the workers are running

One way to check on the workers is to get information from the
ZooKeeper service about how many clients it has.

{% highlight sh %}
{% raw %}
$  echo stat | nc 10.254.139.141 2181; echo
Zookeeper version: 3.4.6--1, built on 10/23/2014 14:18 GMT
Clients:
 /192.168.48.0:44187[0](queued=0,recved=1,sent=0)
 /192.168.45.0:39568[1](queued=0,recved=14072,sent=14072)
 /192.168.86.1:57591[1](queued=0,recved=34,sent=34)
 /192.168.8.0:50375[1](queued=0,recved=34,sent=34)
 /192.168.45.0:39576[1](queued=0,recved=34,sent=34)

Latency min/avg/max: 0/2/2570
Received: 23199
Sent: 23198
Connections: 5
Outstanding: 0
Zxid: 0xa39
Mode: standalone
Node count: 13
{% endraw %}
{% endhighlight %}

There should be one client from the Nimbus service and one per
worker. Ideally, you should get ```stat``` output from ZooKeeper
before and after creating the replication controller.

(Pull requests welcome for alternative ways to validate the workers)

## tl;dr

```kubectl create -f zookeeper.json```

```kubectl create -f zookeeper-service.json```

Make sure the ZooKeeper Pod is running (use: ```kubectl get pods```).

```kubectl create -f storm-nimbus.json```

```kubectl create -f storm-nimbus-service.json```

Make sure the Nimbus Pod is running.

```kubectl create -f storm-worker-controller.json```


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storm/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

