---
layout: docwithnav
title: "title: \"</strong>\""
---
---
layout: docwithnav
title: "</strong>"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Vitess Example

This example shows how to run a [Vitess](http://vitess.io) cluster in Kubernetes.
Vitess is a MySQL clustering system developed at YouTube that makes sharding
transparent to the application layer. It also makes scaling MySQL within
Kubernetes as simple as launching more pods.

The example brings up a database with 2 shards, and then runs a pool of
[sharded guestbook](https://github.com/youtube/vitess/tree/master/examples/kubernetes/guestbook)
pods. The guestbook app was ported from the original
[guestbook](../../examples/guestbook-go/)
example found elsewhere in this tree, modified to use Vitess as the backend.

For a more detailed, step-by-step explanation of this example setup, see the
[Vitess on Kubernetes](http://vitess.io/getting-started/) guide.

### Prerequisites

You'll need to install [Go 1.4+](https://golang.org/doc/install) to build
`vtctlclient`, the command-line admin tool for Vitess.

We also assume you have a running Kubernetes cluster with `kubectl` pointing to
it by default. See the [Getting Started guides](../../docs/getting-started-guides/)
for how to get to that point. Note that your Kubernetes cluster needs to have
enough resources (CPU+RAM) to schedule all the pods. By default, this example
requires a cluster-wide total of at least 6 virtual CPUs and 10GiB RAM. You can
tune these requirements in the
[resource limits](../../docs/user-guide/compute-resources.html)
section of each YAML file.

Lastly, you need to open ports 30000 (for the Vitess admin daemon) and 80 (for
the guestbook app) in your firewall. See the
[Services and Firewalls](../../docs/user-guide/services-firewalls.html)
guide for examples of how to do that.

### Start Vitess

{% highlight console %}
{% raw %}
./vitess-up.sh
{% endraw %}
{% endhighlight %}

This will run through the steps to bring up Vitess. At the end, you should see
something like this:

{% highlight console %}
{% raw %}
****************************
* Complete!
* Use the following line to make an alias to kvtctl:
* alias kvtctl='$GOPATH/bin/vtctlclient -server 104.197.47.173:30000'
* See the vtctld UI at: http://104.197.47.173:30000
****************************
{% endraw %}
{% endhighlight %}

### Start the Guestbook app

{% highlight console %}
{% raw %}
./guestbook-up.sh
{% endraw %}
{% endhighlight %}

The guestbook service is configured with `type: LoadBalancer` to tell Kubernetes
to expose it on an external IP. It may take a minute to set up, but you should
soon see the external IP show up under the internal one like this:

{% highlight console %}
{% raw %}
$ kubectl get service guestbook
NAME        LABELS    SELECTOR         IP(S)             PORT(S)
guestbook   <none>    name=guestbook   10.67.253.173     80/TCP
                                       104.197.151.132
{% endraw %}
{% endhighlight %}

Visit the external IP in your browser to view the guestbook. Note that in this
modified guestbook, there are multiple pages to demonstrate range-based sharding
in Vitess. Each page number is assigned to one of the shards using a
[consistent hashing](https://en.wikipedia.org/wiki/Consistent_hashing) scheme.

### Tear down

{% highlight console %}
{% raw %}
./guestbook-down.sh
./vitess-down.sh
{% endraw %}
{% endhighlight %}

You may also want to remove any firewall rules you created.

### Limitations

Currently this example cluster is not configured to use the built-in
[Backup/Restore](http://vitess.io/user-guide/backup-and-restore.html) feature of
Vitess, because as of
[Vitess v2.0.0-alpha2](https://github.com/youtube/vitess/releases) that feature
requires a network-mounted directory. Usually this system is used to restore
from the latest backup when a pod is moved or added in an existing deployment.
As part of the final Vitess v2.0.0 release, we plan to provide support for
saving backups in a cloud-based blob store (such as Google Cloud Storage or
Amazon S3), which we believe will be better suited to running in Kubernetes.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/vitess/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->


