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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/config-best-practices.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Tips and tricks when working with config

This document is meant to highlight and consolidate in one place configuration best practices that are introduced throughout the user-guide and getting-started documentation and examples. This is a living document so if you think of something that is not on this list but might be useful to others, please don't hesitate to file an issue or submit a PR.

1. When writing configuration, use the latest stable API version (currently v1).
1. Configuration should be stored in version control before being pushed to the cluster. This allows configuration to be quickly rolled back if needed and will aid with cluster re-creation and restoration if the worst were to happen.
1. Use YAML rather than JSON. They can be used interchangeably in almost all scenarios but YAML tends to be more user-friendly for config.
1. Group related objects together in a single file. This is often better than separate files.
1. Use `kubectl create -f <directory>` where possible. This looks for config objects in all `.yaml`, `.yml`, and `.json` files in `<directory>` and passes them to create.
1. Create a service before corresponding replication controllers so that the scheduler can spread the pods comprising the service. You can also create the replication controller without specifying replicas, create the service, then scale up the replication controller, which may work better in an example using progressive disclosure and may have benefits in real scenarios also, such as ensuring one replica works before creating lots of them)
1. Don't use `hostPort` unless absolutely necessary (e.g., for a node daemon) as it will prevent certain scheduling configurations due to port conflicts. Use the apiserver proxying or port forwarding for debug/admin access, or a service for external service access. If you need to expose a pod's port on the host machine, consider using a [NodePort](services.md#type--loadbalancer) service before resorting to `hostPort`. If you only need access to the port for debugging purposes, you can also use the [kubectl proxy and apiserver proxy](connecting-to-applications-proxy.md) or [kubectl port-forward](connecting-to-applications-port-forward.md).
1. Don't use `hostNetwork` for the same reasons as `hostPort`.
1. Don't specify default values unnecessarily, to simplify and minimize configs. For example, omit the selector and labels in ReplicationController if you want them to be the same as the labels in its podTemplate, since those fields are populated from the podTemplate labels by default.
1. Instead of attaching one label to a set of pods to represent a service (e.g., `service: myservice`) and another to represent the replication controller managing the pods (e.g., `controller: mycontroller`), attach labels that identify semantic attributes of your application or deployment and select the appropriate subsets in your service and replication controller, such as `{ app: myapp, tier: frontend, deployment: v3 }`. A service can be made to span multiple deployments, such as across rolling updates, by simply omitting release-specific labels from its selector, rather than updating a service's selector to match the replication controller's selector fully.
1. Use kubectl bulk operations (via files and/or labels) for get and delete. See [label selectors](labels.md#label-selectors) and [using labels effectively](managing-deployments.md#using-labels-effectively).
1. Use kubectl run and expose to quickly create and expose single container replication controllers. See the [quick start guide](quick-start.md) for an example.
1. Use headless services for easy service discovery when you don't need kube-proxy load balancing. See [headless services](services.md#headless-services).
1. Use kubectl delete rather than stop. Delete has a superset of the functionality of stop and stop is deprecated.
1. If there is a viable alternative to naked pods (i.e. pods not bound to a controller), go with the alternative. Controllers are almost always preferable to creating pods (except for some `restartPolicy: Never` scenarios). A minimal Job is coming. See [#1624](https://github.com/GoogleCloudPlatform/kubernetes/issues/1624). Naked pods will not be rescheduled in the event of node failure.
1. Put a version number or hash as a suffix to the name and in a label on a replication controller to facilitate rolling update, as we do for [--image](kubectl/kubectl_rolling-update.md). This is necessary because rolling-update actually creates a new controller as opposed to modifying the existing controller. This does not play well with version agnostic controller names.
1. Put an object description in an annotation to allow better introspection.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/config-best-practices.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
