<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Prerequisites

To deploy and manage applications on Kubernetes, you’ll use the Kubernetes command-line tool, [kubectl](../../docs/kubectl.md). It can be found in the release tar bundle, or can be built from source from github. Ensure that it is executable and in your path.

In order for kubectl to find and access the Kubernetes cluster, it needs a [kubeconfig file](../../docs/kubeconfig-file.md), which is created automatically when creating a cluster using kube-up.sh (see the [getting started guides](../../docs/getting-started-guides/) for more about creating clusters). If you need access to a cluster you didn’t create, see the [Sharing Cluster Access document](../../docs/sharing-clusters.md).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/prereqs.md?pixel)]()
