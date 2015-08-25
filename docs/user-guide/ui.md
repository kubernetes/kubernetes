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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/ui.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Interface

Kubernetes has a web-based user interface that displays the current cluster state graphically.

## Accessing the UI

By default, the Kubernetes UI is deployed as a cluster addon. To access it, visit `https://<kubernetes-master>/ui`, which redirects to `https://<kubernetes-master>/api/v1/proxy/namespaces/kube-system/services/kube-ui/#/dashboard/`.

If you find that you're not able to access the UI, it may be because the kube-ui service has not been started on your cluster. In that case, you can start it manually with:

```sh
kubectl create -f cluster/addons/kube-ui/kube-ui-rc.yaml --namespace=kube-system
kubectl create -f cluster/addons/kube-ui/kube-ui-svc.yaml --namespace=kube-system
```

Normally, this should be taken care of automatically by the [`kube-addons.sh`](http://releases.k8s.io/HEAD/cluster/saltbase/salt/kube-addons/kube-addons.sh) script that runs on the master.

## Using the UI

The Kubernetes UI can be used to introspect your current cluster, such as checking how resources are used, or looking at error messages. You cannot, however, use the UI to modify your cluster.

### Node Resource Usage

After accessing Kubernetes UI, you'll see a homepage dynamically listing out all nodes in your current cluster, with related information including internal IP addresses, CPU usage, memory usage, and file systems usage.
![Kubernetes UI home page](k8s-ui-overview.png)

### Dashboard Views

Click on the "Views" button in the top-right of the page to see other views available, which include: Explore, Pods, Nodes, Replication Controllers, Services, and Events.

#### Explore View

The "Explore" view allows your to see the pods, replication controllers, and services in current cluster easily.
![Kubernetes UI Explore View](k8s-ui-explore.png)
The "Group by" dropdown list allows you to group these resources by a number of factors, such as type, name, host, etc.
![Kubernetes UI Explore View - Group by](k8s-ui-explore-groupby.png)
You can also create filters by clicking on the down triangle of any listed resource instances and choose which filters you want to add.
![Kubernetes UI Explore View - Filter](k8s-ui-explore-filter.png)
To see more details of each resource instance, simply click on it.
![Kubernetes UI - Pod](k8s-ui-explore-poddetail.png)

### Other Views

Other views (Pods, Nodes, Replication Controllers, Services, and Events) simply list information about each type of resource. You can also click on any instance for more details.
![Kubernetes UI - Nodes](k8s-ui-nodes.png)

## More Information

For more information, see the [Kubernetes UI development document](http://releases.k8s.io/HEAD/www/README.md) in the www directory.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/ui.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
