<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Supporting multiple API groups

## Goal

1. Breaking the monolithic v1 API into modular groups and allowing groups to be enabled/disabled individually. This allows us to break the monolithic API server to smaller components in the future.

2. Supporting different versions in different groups. This allows different groups to evolve at different speed.

3. Supporting identically named kinds to exist in different groups. This is useful when we experiment new features of an API in the experimental group while supporting the stable API in the original group at the same time.

4. Exposing the API groups and versions supported by the server. This is required to develop a dynamic client.

5. Laying the basis for [API Plugin](../../docs/design/extending-api.md).

6. Keeping the user interaction easy. For example, we should allow users to omit group name when using kubectl if there is no ambiguity.


## Bookkeeping for groups

1. No changes to TypeMeta:

  Currently many internal structures, such as RESTMapper and Scheme, are indexed and retrieved by APIVersion. For a fast implementation targeting the v1.1 deadline, we will concatenate group with version, in the form of "group/version", and use it where a version string is expected, so that many code can be reused. This implies we will not add a new field to TypeMeta, we will use TypeMeta.APIVersion to hold "group/version".

  For backward compatibility, v1 objects belong to the group with an empty name, so existing v1 config files will remain valid.

2. /pkg/conversion#Scheme:

  The key of /pkg/conversion#Scheme.versionMap for versioned types will be "group/version". For now, the internal version types of all groups will be registered to versionMap[""], as we don't have any identically named kinds in different groups yet. In the near future, internal version types will be registered to versionMap["group/"], and pkg/conversion#Scheme.InternalVersion will have type []string.

  We will need a mechanism to express if two kinds in different groups (e.g., compute/pods and experimental/pods) are convertible, and auto-generate the conversions if they are.

3. meta.RESTMapper:

  Each group will have its own RESTMapper (of type DefaultRESTMapper), and these mappers will be registered to pkg/api#RESTMapper (of type MultiRESTMapper).

  To support identically named kinds in different groups, We need to expand the input of RESTMapper.VersionAndKindForResource from (resource string) to (group, resource string). If group is not specified and there is ambiguity (i.e., the resource exists in multiple groups), an error should be returned to force the user to specify the group.

## Server-side implementation

1. resource handlers' URL:

  We will force the URL to be in the form of prefix/group/version/...

  Prefix is used to differentiate API paths from other paths like /healthz. All groups will use the same prefix="apis", except when backward compatibility requires otherwise. No "/" is allowed in prefix, group, or version. Specifically,

    * for /api/v1, we set the prefix="api" (which is populated from cmd/kube-apiserver/app#APIServer.APIPrefix), group="", version="v1", so the URL remains to be /api/v1.

    * for new kube API groups, we will set the prefix="apis" (we will add a field in type APIServer to hold this prefix), group=GROUP_NAME, version=VERSION. For example, the URL of the experimental resources will be /apis/experimental/v1alpha1.

    * for OpenShift v1 API, because it's currently registered at /oapi/v1, to be backward compatible, OpenShift may set prefix="oapi", group="".

    * for other new third-party API, they should also use the prefix="apis" and choose the group and version. This can be done through the thirdparty API plugin mechanism in [13000](http://pr.k8s.io/13000).

2. supporting API discovery:

  * At /prefix (e.g., /apis), API server will return the supported groups and their versions using pkg/api/unversioned#APIVersions type, setting the Versions field to "group/version". This is backward compatible, because currently API server does return "v1" encoded in pkg/api/unversioned#APIVersions at /api. (We will also rename the JSON field name from `versions` to `apiVersions`, to be consistent with pkg/api#TypeMeta.APIVersion field)

  * At /prefix/group, API server will return all supported versions of the group. We will create a new type VersionList (name is open to discussion) in pkg/api/unversioned as the API.

  * At /prefix/group/version, API server will return all supported resources in this group, and whether each resource is namespaced. We will create a new type APIResourceList (name is open to discussion) in pkg/api/unversioned as the API.

  We will design how to handle deeper path in other proposals.

  * At /swaggerapi/swagger-version/prefix/group/version, API server will return the Swagger spec of that group/version in `swagger-version` (e.g. we may support both Swagger v1.2 and v2.0).

3. handling common API objects:

  * top-level common API objects:

    To handle the top-level API objects that are used by all groups, we either have to register them to all schemes, or we can choose not to encode them to a version. We plan to take the latter approach and place such types in a new package called `unversioned`, because many of the common top-level objects, such as APIVersions, VersionList, and APIResourceList, which are used in the API discovery, and pkg/api#Status, are part of the protocol between client and server, and do not belong to the domain-specific parts of the API, which will evolve independently over time.

    Types in the unversioned package will not have the APIVersion field, but may retain the Kind field.

    For backward compatibility, when hanlding the Status, the server will encode it to v1 if the client expects the Status to be encoded in v1, otherwise the server will send the unversioned#Status. If an error occurs before the version can be determined, the server will send the unversioned#Status.

  * non-top-level common API objects:

    Assuming object o belonging to group X is used as a field in an object belonging to group Y, currently genconversion will generate the conversion functions for o in package Y. Hence, we don't need any special treatment for non-top-level common API objects.

    TypeMeta is an exception, because it is a common object that is used by objects in all groups but does not logically belong to any group. We plan to move it to the package `unversioned`.

## Client-side implementation

1. clients:

  Currently we have structured (pkg/client/unversioned#ExperimentalClient, pkg/client/unversioned#Client) and unstructured (pkg/kubectl/resource#Helper) clients. The structured clients are not scalable because each of them implements specific interface, e.g., [here](../../pkg/client/unversioned/client.go#L32). Only the unstructured clients are scalable. We should either auto-generate the code for structured clients or migrate to use the unstructured clients as much as possible.

  We should also move the unstructured client to pkg/client/.

2. Spelling the URL:

  The URL is in the form of prefix/group/version/. The prefix is hard-coded in the client/unversioned.Config. The client should be able to figure out `group` and `version` using the RESTMapper. For a third-party client which does not have access to the RESTMapper, it should discover the mapping of `group`, `version` and `kind` by querying the server as described in point 2 of #server-side-implementation.

3. kubectl:

  kubectl should accept arguments like `group/resource`, `group/resource/name`. Nevertheless, the user can omit the `group`, then kubectl shall rely on RESTMapper.VersionAndKindForResource() to figure out the default group/version of the resource. For example, for resources (like `node`) that exist in both k8s v1 API and k8s modularized API (like `infra/v2`), we should set kubectl default to use one of them. If there is no default group, kubectl should return an error for the ambiguity.

  When kubectl is used with a single resource type, the --api-version and --output-version flag of kubectl should accept values in the form of `group/version`, and they should work as they do today. For multi-resource operations, we will disable these two flags initially.

  Currently, by setting pkg/client/unversioned/clientcmd/api/v1#Config.NamedCluster[x].Cluster.APIVersion ([here](../../pkg/client/unversioned/clientcmd/api/v1/types.go#L58)), user can configure the default apiVersion used by kubectl to talk to server. It does not make sense to set a global version used by kubectl when there are multiple groups, so we plan to deprecate this field. We may extend the version negotiation function to negotiate the preferred version of each group. Details will be in another proposal.

## OpenShift integration

OpenShift can take a similar approach to break monolithic v1 API: keeping the v1 where they are, and gradually adding groups.

For the v1 objects in OpenShift, they should keep doing what they do now: they should remain registered to Scheme.versionMap["v1"] scheme, they should keep being added to originMapper.

For new OpenShift groups, they should do the same as native Kubernetes groups would do: each group should register to Scheme.versionMap["group/version"], each should has separate RESTMapper and the register the MultiRESTMapper.

To expose a list of the supported Openshift groups to clients, OpenShift just has to call to pkg/cmd/server/origin#call initAPIVersionRoute() as it does now, passing in the supported "group/versions" instead of "versions".


## Future work

1. Dependencies between groups: we need an interface to register the dependencies between groups. It is not our priority now as the use cases are not clear yet.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/api-group.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
