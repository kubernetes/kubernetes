---
layout: docwithnav
title: "Kubernetes User Guide: Managing Applications: Prerequisites"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Prerequisites

To deploy and manage applications on Kubernetes, you’ll use the Kubernetes command-line tool, [kubectl](kubectl/kubectl.html). It lets you inspect your cluster resources, create, delete, and update components, and much more. You will use it to look at your new cluster and bring up example apps. 

## Install kubectl

You can find it in the [release](https://github.com/GoogleCloudPlatform/kubernetes/releases) tar bundle, under platforms/<os>/<arch>;
or if you build from source, kubectl should be either under _output/local/bin/<os>/<arch> or _output/dockerized/bin/<os>/<arch>.

Next, make sure the kubectl tool is in your path, assuming you download a release:

{% highlight bash %}
{% raw %}
# OS X
export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

# Linux
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
{% endraw %}
{% endhighlight %}

## Configure kubectl

In order for kubectl to find and access the Kubernetes cluster, it needs a [kubeconfig file](kubeconfig-file.html), which is created automatically when creating a cluster using kube-up.sh (see the [getting started guides](../../docs/getting-started-guides/) for more about creating clusters). If you need access to a cluster you didn’t create, see the [Sharing Cluster Access document](sharing-clusters.html).

#### Installing Kubectl

If you downloaded a pre-compiled release, kubectl should be under `platforms/<os>/<arch>`.

If you built from source, kubectl should be either under `_output/local/bin/<os>/<arch>` or `_output/dockerized/bin/<os>/<arch>`.

The kubectl binary doesn't have to be installed to be executable, but the rest of the walkthrough will assume that it's in your PATH.

The simplest way to install is to copy or move kubectl into a dir already in PATH (e.g. `/usr/local/bin`). For example:

{% highlight console %}
{% raw %}
# OS X
$ sudo cp kubernetes/platforms/darwin/amd64/kubectl /usr/local/bin/kubectl
# Linux
$ sudo cp kubernetes/platforms/linux/amd64/kubectl /usr/local/bin/kubectl
{% endraw %}
{% endhighlight %}

#### Configuring Kubectl

If you used `./cluster/kube-up.sh` to deploy your Kubernetes cluster, kubectl should already be locally configured.

By default, kubectl configuration lives at `~/.kube/config`.

If your cluster was deployed by other means (e.g. a [getting started guide](../getting-started-guides/README.html)) your kubectl client will typically be configured during that process. If for some reason your kubectl client is not yet configured, check out [kubeconfig-file.md](kubeconfig-file.html).

#### Making sure you're ready

Check that kubectl is properly configured by getting the cluster state:

{% highlight console %}
{% raw %}
$ kubectl cluster-info
{% endraw %}
{% endhighlight %}

If you see a url response, you are ready to go.

## What's next?

[Learn how to launch and expose your application.](quick-start.html)


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/prereqs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

