<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Prerequisites
To deploy and manage applications on Kubernetes, you’ll use the Kubernetes command-line tool, [kubectl](kubectl/kubectl.md). It lets you inspect your cluster resources, create, delete, and update components, and much more. You will use it to look at your new cluster and bring up example apps. 

##Install kubectl
You can find it in the [release](https://github.com/GoogleCloudPlatform/kubernetes/releases) tar bundle, under platforms/<os>/<arch>;
or if you build from source, kubectl should be either under _output/local/bin/<os>/<arch> or _output/dockerized/bin/<os>/<arch>.

Next, make sure the kubectl tool is in your path, assuming you download a release:
```
# OS X
export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

# Linux
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
```

##Configure kubectl
In order for kubectl to find and access the Kubernetes cluster, it needs a [kubeconfig file](kubeconfig-file.md), which is created automatically when creating a cluster using kube-up.sh (see the [getting started guides](../../docs/getting-started-guides/) for more about creating clusters). If you need access to a cluster you didn’t create, see the [Sharing Cluster Access document](sharing-clusters.md).

Check that kubectl is properly configured by getting the cluster state:
```
$ kubectl cluster-info
```
## What's next?

[Learn how to launch and expose your application.](quick-start.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/prereqs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
