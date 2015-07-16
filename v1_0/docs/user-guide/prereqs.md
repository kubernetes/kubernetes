---
layout: docwithnav
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Prerequisites

To deploy and manage applications on Kubernetes, you’ll use the Kubernetes command-line tool, [kubectl](kubectl/kubectl.md). It can be found in the release tar bundle, or can be built from source from github. Ensure that it is executable and in your path.

In order for kubectl to find and access the Kubernetes cluster, it needs a [kubeconfig file](kubeconfig-file.md), which is created automatically when creating a cluster using kube-up.sh (see the [getting started guides](../../docs/getting-started-guides/) for more about creating clusters). If you need access to a cluster you didn’t create, see the [Sharing Cluster Access document](sharing-clusters.md).

## What's next?

[Learn how to launch and expose your application.](quick-start.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/prereqs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
