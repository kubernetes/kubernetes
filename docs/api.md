# The Kubernetes API

Primary system and API concepts are documented in the [User guide](user-guide.md).

Overall API conventions are described in the [API conventions doc](api-conventions.md).

Complete API details are documented via [Swagger](http://swagger.io/). The Kubernetes apiserver (aka "master") exports an API that can be used to retrieve the [Swagger spec](https://github.com/swagger-api/swagger-spec/tree/master/schemas/v1.2) for the Kubernetes API, by default at `/swaggerapi`, and a UI you can use to browse the API documentation at `/swagger-ui`. We also periodically update a [statically generated UI](http://kubernetes.io/third_party/swagger-ui/).

Remote access to the API is discussed in the [access doc](accessing_the_api.md).

The Kubernetes API also serves as the foundation for the declarative configuration schema for the system. The [Kubectl](kubectl.md) command-line tool can be used to create, update, delete, and get API objects.

Kubernetes also stores its serialized state (currently in [etcd](https://coreos.com/docs/distributed-configuration/getting-started-with-etcd/)) in terms of the API resources.

Kubernetes itself is decomposed into multiple components, which interact through its API.

## API changes

In our experience, any system that is successful needs to grow and change as new use cases emerge or existing ones change. Therefore, we expect the Kubernetes API to continuously change and grow. However, we intend to not break compatibility with existing clients, for an extended period of time. In general, new API resources and new resource fields can be expected to be added frequently. Elimination of resources or fields will require following a deprecation process. The precise deprecation policy for eliminating features is TBD, but once we reach our 1.0 milestone, there will be a specific policy.

What constitutes a compatible change and how to change the API are detailed by the [API change document](devel/api_changes.md).

## API versioning

Fine-grain resource evolution alone makes it difficult to eliminate fields or restructure resource representations. Therefore, Kubernetes supports multiple API versions, each at a different API path prefix, such as `/api/v1beta3`. These are simply different interfaces to read and/or modify the same underlying resources. In general, all API resources are accessible via all API versions, though there may be some cases in the future where that is not true. 

Distinct API versions present more clear, consistent views of system resources and behavior than intermingled, independently evolved resources. They also provide a more straightforward mechanism for controlling access to end-of-lifed and/or experimental APIs.

The [API and release versioning proposal](versioning.md) describes the current thinking on the API version evolution process.

## v1beta1 and v1beta2 are deprecated; please move to v1beta3 ASAP

As of April 1, 2015, the Kubernetes v1beta3 API has been enabled by default, and the v1beta1 and v1beta2 APIs are deprecated. v1beta3 should be considered the v1 release-candidate API, and the v1 API is expected to be substantially similar. As "pre-release" APIs, v1beta1, v1beta2, and v1beta3 will be eliminated once the v1 API is available, by the end of June 2015. 

## v1beta3 conversion tips

We're working to convert all documentation and examples to v1beta3. Most examples already contain a v1beta3 subdirectory with the API objects translated to v1beta3. A simple [API conversion tool](cluster_management.md#switching-your-config-files-to-a-new-api-version) has been written to simplify the translation process. Use `kubectl create --validate` in order to validate your json or yaml against our Swagger spec. 

Some important differences between v1beta1/2 and v1beta3:
* The resource `id` is now called `name`.
* `name`, `labels`, `annotations`, and other metadata are now nested in a map called `metadata`
* `desiredState` is now called `spec`, and `currentState` is now called `status`
* `/minions` has been moved to `/nodes`, and the resource has kind `Node`
* The namespace is required (for all namespaced resources) and has moved from a URL parameter to the path: `/api/v1beta3/namespaces/{namespace}/{resource_collection}/{resource_name}`
* The names of all resource collections are now lower cased - instead of `replicationControllers`, use `replicationcontrollers`.
* To watch for changes to a resource, open an HTTP or Websocket connection to the collection query and provide the `?watch=true` query parameter along with the desired `resourceVersion` parameter to watch from.
* The `labels` query parameter has been renamed to `label-selector`.
* The container `entrypoint` has been renamed to `command`, and `command` has been renamed to `args`.
* Container, volume, and node resources are expressed as nested maps (e.g., `resources{cpu:1}`) rather than as individual fields, and resource values support [scaling suffixes](resources.md#resource-quantities) rather than fixed scales (e.g., milli-cores).
* Restart policy is represented simply as a string (e.g., `"Always"`) rather than as a nested map (`always{}`).
* Pull policies changed from `PullAlways`, `PullNever`, and `PullIfNotPresent` to `Always`, `Never`, and `IfNotPresent`.
* The volume `source` is inlined into `volume` rather than nested.
* Host volumes have been changed from `hostDir` to `hostPath` to better reflect that they can be files or directories.
