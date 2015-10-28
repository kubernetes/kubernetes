<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

To make it easier to eliminate fields or restructure resource representations, Kubernetes supports
multiple API versions, each at a different API path, such as `/api/v1` or
`/apis/extensions/v1beta1`.

We chose to version at the API level rather than at the resource or field level to ensure that the API presents a clear, consistent view of system resources and behavior, and to enable controlling access to end-of-lifed and/or experimental APIs.

Note that API versioning and Software versioning are only indirectly related.  The [API and release
versioning proposal](design/versioning.md) describes the relationship between API versioning and
software versioning.


Different API versions imply different levels of stability and support.  The criteria for each level are described
in more detail in the [API Changes documentation](devel/api_changes.md#alpha-beta-and-stable-versions).  They are summarized here:

- Alpha level:
  - The version names contain `alpha` (e.g. `v1alpha1`).
  - May be buggy.  Enabling the feature may expose bugs.  Disabled by default.
  - Support for feature may be dropped at any time without notice.
  - The API may change in incompatible ways in a later software release without notice.
  - Recommended for use only in short-lived testing clusters, due to increased risk of bugs and lack of long-term support.
- Beta level:
  - The version names contain `beta` (e.g. `v2beta3`).
  - Code is well tested.  Enabling the feature is considered safe.  Enabled by default.
  - Support for the overall feature will not be dropped, though details may change.
  - The schema and/or semantics of objects may change in incompatible ways in a subsequent beta or stable release.  When this happens,
    we will provide instructions for migrating to the next version.  This may require deleting, editing, and re-creating
    API objects.  The editing process may require some thought.   This may require downtime for appplications that rely on the feature.
  - Recommended for only non-business-critical uses because of potential for incompatible changes in subsequent releases.  If you have
    multiple clusters which can be upgraded independently, you may be able to relax this restriction.
  - **Please do try our beta features and give feedback on them!  Once they exit beta, it may not be practical for us to make more changes.**
- Stable level:
  - The version name is `vX` where `X` is an integer.
  - Stable versions of features will appear in released software for many subsequent versions.

## API groups

To make it easier to extend the Kubernetes API, we are in the process of implementing [*API
groups*](proposals/api-groups.md).  These are simply different interfaces to read and/or modify the
same underlying resources.  The API group is specified in a REST path and in the `apiVersion` field
of a serialized object.

Currently there are two API groups in use:

1. the "core" group, which is at REST path `/api/v1` and is not specified as part of the `apiVersion` field, e.g.
   `apiVersion: v1`.
1. the "extensions" group, which is at REST path `/apis/extensions/$VERSION`, and which uses
  `apiVersion: extensions/$VERSION` (e.g. currently `apiVersion: extensions/v1beta1`).

In the future we expect that there will be more API groups, all at REST path `/apis/$API_GROUP` and
using `apiVersion: $API_GROUP/$VERSION`.  We expect that there will be a way for (third parties to
create their own API groups](design/extending-api.md), and to avoid naming collisions.

## Enabling the extensions group

Enable `extensions/v1beta1` objects by adding the following flags to your API server:

  - `--runtime-config=extensions/v1beta1=true`

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




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
