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
[here](http://releases.k8s.io/release-1.0/docs/api.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# The Kubernetes API

Primary system and API concepts are documented in the [User guide](user-guide/README.md).

Overall API conventions are described in the [API conventions doc](devel/api-conventions.md).

Complete API details are documented via [Swagger](http://swagger.io/). The Kubernetes apiserver (aka "master") exports an API that can be used to retrieve the [Swagger spec](https://github.com/swagger-api/swagger-spec/tree/master/schemas/v1.2) for the Kubernetes API, by default at `/swaggerapi`, and a UI you can use to browse the API documentation at `/swagger-ui`. We also periodically update a [statically generated UI](http://kubernetes.io/third_party/swagger-ui/).

Remote access to the API is discussed in the [access doc](admin/accessing-the-api.md).

The Kubernetes API also serves as the foundation for the declarative configuration schema for the system. The [Kubectl](user-guide/kubectl/kubectl.md) command-line tool can be used to create, update, delete, and get API objects.

Kubernetes also stores its serialized state (currently in [etcd](https://coreos.com/docs/distributed-configuration/getting-started-with-etcd/)) in terms of the API resources.

Kubernetes itself is decomposed into multiple components, which interact through its API.

## API changes

In our experience, any system that is successful needs to grow and change as new use cases emerge or existing ones change. Therefore, we expect the Kubernetes API to continuously change and grow. However, we intend to not break compatibility with existing clients, for an extended period of time. In general, new API resources and new resource fields can be expected to be added frequently. Elimination of resources or fields will require following a deprecation process. The precise deprecation policy for eliminating features is TBD, but once we reach our 1.0 milestone, there will be a specific policy.

What constitutes a compatible change and how to change the API are detailed by the [API change document](devel/api_changes.md).

## API versioning

To make it easier to eliminate fields or restructure resource representations, Kubernetes supports multiple API versions, each at a different API path prefix, such as `/api/v1beta3`. These are simply different interfaces to read and/or modify the same underlying resources. In general, all API resources are accessible via all API versions, though there may be some cases in the future where that is not true.

We chose to version at the API level rather than at the resource or field level to ensure that the API presents a clear, consistent view of system resources and behavior, and to enable controlling access to end-of-lifed and/or experimental APIs.

The [API and release versioning proposal](design/versioning.md) describes the current thinking on the API version evolution process.

## v1beta1, v1beta2, and v1beta3 are deprecated; please move to v1 ASAP

As of June 4, 2015, the Kubernetes v1 API has been enabled by default. The v1beta1 and v1beta2 APIs were deleted on June 1, 2015. v1beta3 is planned to be deleted on July 6, 2015.

### v1 conversion tips (from v1beta3)

We're working to convert all documentation and examples to v1. A simple [API conversion tool](admin/cluster-management.md#switching-your-config-files-to-a-new-api-version) has been written to simplify the translation process. Use `kubectl create --validate` in order to validate your json or yaml against our Swagger spec.

Changes to services are the most significant difference between v1beta3 and v1.

* The `service.spec.portalIP` property is renamed to `service.spec.clusterIP`.
* The `service.spec.createExternalLoadBalancer` property is removed. Specify `service.spec.type: "LoadBalancer"` to create an external load balancer instead.
* The `service.spec.publicIPs` property is deprecated and now called `service.spec.deprecatedPublicIPs`. This property will be removed entirely when v1beta3 is removed. The vast majority of users of this field were using it to expose services on ports on the node. Those users should specify `service.spec.type: "NodePort"` instead. Read [External Services](user-guide/services.md#external-services) for more info. If this is not sufficient for your use case, please file an issue or contact @thockin.

Some other difference between v1beta3 and v1:

* The `pod.spec.containers[*].privileged` and `pod.spec.containers[*].capabilities` properties are now nested under the `pod.spec.containers[*].securityContext` property. See [Security Contexts](user-guide/security-context.md).
* The `pod.spec.host` property is renamed to `pod.spec.nodeName`.
* The `endpoints.subsets[*].addresses.IP` property is renamed to `endpoints.subsets[*].addresses.ip`.
* The `pod.status.containerStatuses[*].state.termination` and `pod.status.containerStatuses[*].lastState.termination` properties are renamed to `pod.status.containerStatuses[*].state.terminated` and `pod.status.containerStatuses[*].lastState.terminated` respectively.
* The `pod.status.Condition` property is renamed to `pod.status.conditions`.
* The `status.details.id` property is renamed to `status.details.name`.

### v1beta3 conversion tips (from v1beta1/2)

Some important differences between v1beta1/2 and v1beta3:

* The resource `id` is now called `name`.
* `name`, `labels`, `annotations`, and other metadata are now nested in a map called `metadata`
* `desiredState` is now called `spec`, and `currentState` is now called `status`
* `/minions` has been moved to `/nodes`, and the resource has kind `Node`
* The namespace is required (for all namespaced resources) and has moved from a URL parameter to the path: `/api/v1beta3/namespaces/{namespace}/{resource_collection}/{resource_name}`. If you were not using a namespace before, use `default` here.
* The names of all resource collections are now lower cased - instead of `replicationControllers`, use `replicationcontrollers`.
* To watch for changes to a resource, open an HTTP or Websocket connection to the collection query and provide the `?watch=true` query parameter along with the desired `resourceVersion` parameter to watch from.
* The `labels` query parameter has been renamed to `labelSelector`.
* The `fields` query parameter has been renamed to `fieldSelector`.
* The container `entrypoint` has been renamed to `command`, and `command` has been renamed to `args`.
* Container, volume, and node resources are expressed as nested maps (e.g., `resources{cpu:1}`) rather than as individual fields, and resource values support [scaling suffixes](user-guide/compute-resources.md#specifying-resource-quantities) rather than fixed scales (e.g., milli-cores).
* Restart policy is represented simply as a string (e.g., `"Always"`) rather than as a nested map (`always{}`).
* Pull policies changed from `PullAlways`, `PullNever`, and `PullIfNotPresent` to `Always`, `Never`, and `IfNotPresent`.
* The volume `source` is inlined into `volume` rather than nested.
* Host volumes have been changed from `hostDir` to `hostPath` to better reflect that they can be files or directories.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
