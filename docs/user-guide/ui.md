<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/user-guide/ui.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Dashboard User Interface

Kubernetes has a web-based user interface that allows users to manage applications running in
the cluster and troubleshoot them, as well as manage the cluster itself.

## Accessing the Dashboard

By default, the Kubernetes Dashboard is deployed as a cluster addon. To access it, visit
`https://<kubernetes-master>/ui`, which redirects to
`https://<kubernetes-master>/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard`.

If you find that you're not able to access the Dashboard, it may be because the kubernetes-dashboard
service has not been started on your cluster. In that case, you can start it manually with:

```sh
kubectl create -f cluster/addons/dashboard/dashboard-controller.yaml --namespace=kube-system
kubectl create -f cluster/addons/dashboard/dashboard-service.yaml --namespace=kube-system
```

Normally, this should be taken care of automatically by the
[`kube-addons.sh`](http://releases.k8s.io/HEAD/cluster/saltbase/salt/kube-addons/kube-addons.sh)
script that runs on the master. Release notes and development versions of the Dashboard can be
found at https://github.com/kubernetes/dashboard/releases.

## Overview

The Dashboard can be used to provide an overview of applications running on the cluster and provide
information on any errors that have occurred. You can also inspect your replication controllers and
corresponding services, change the number of replicas and deploy new applications using a wizard.

### Using the Dashboard

When first accessing the dashboard on an empty cluster, you should see the welcome page.
This contains some useful links to the documentation, and a big button to deploy your first
application.
![Kubernetes Dashboard welcome page](ui-dashboard-zerostate.png)

### Deploying applications

The Kubernetes Dashboard lets you create and deploy a Replication Controller with a simple wizard.
You can simply provide the name for your application, the name of a Docker container (commonly
hosted on the Google Container Registry or Docker Hub) and the target number of Pods you want deployed.
Optionally, if your container listens on a port, you can also provide a port and target port. The
wizard will create a corresponding Kubernetes Service which will route to your deployed Pods.

![Kubernetes Dashboard deploy form](ui-dashboard-deploy-simple.png)

If needed, you can expand the "more options" section where you can change more advanced settings,
such as the Kubernetes namespace that the resulting Pods run in, image pull secrets for private
registries, resource limits, container entrypoint and privileged status.

![Kubernetes Dashboard deploy form advanced options](ui-dashboard-deploy-more.png)

#### Applications view

If some applications are running on your cluster, the Dashboard will default to showing an overview.
Individual applications are shown as cards - where an application is defined as a Replication Controller
and its corresponding services. Each card shows the current number of replicas running and desired,
along with any errors reported by Kubernetes. You can also view logs, make quick changes to the number
of replicas or delete the application directly from the menu in the cards' corner.

![Kubernetes Dashboard applications view](ui-dashboard-rcs.png)

Clicking "View details" from the card menu will take you to the following screen, where you
can view more information about the Pods that make up your application. The events tab can be useful
in debugging flapping applications.

![Kubernetes Dashboard application detail](ui-dashboard-rcs-detail.png)

## More Information

For more information, see the
[Kubernetes Dashboard repository](https://github.com/kubernetes/dashboard).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/ui.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
