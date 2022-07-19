# Why this forked repository ?

This repository carries the prototype branch which accumulates the hacks, prototypes, proto-KEP experiments, and workarounds required to make [KCP](https://github.com/kcp-dev/kcp/blob/main/README.md) a reality.
It is based on K8S 1.22 for now and commits are identified with basic labels like HACK/FEATURE/WORKAROUND.

# Summary of changes

The detailed explanation of the changes made on top of the Kubernetes code can be found in both the commit messages, and comments of the associated code.

However here is a summary of the changes, with the underlying requirements and motivations. Reading the linked investigation document first will help.

## A. Minimal API Server

__Investigation document:__ [minimal-api-server.md](https://github.com/kcp-dev/kcp/blob/main/docs/investigations/minimal-api-server.md)

1.  New generic control plane based on kube api-server

    It is mainly provided by code:
    
    1. initially duplicated from kube api-server main code, and legacy scheme (`DUPLICATE` commits),
        
    2. then stripped down from unnecessary things (ergress, api aggregation, webhooks) and APIs (Pods, Nodes, Deployments, etc ...) (`NEW` commits)

2.  Support adding K8S built-in resources (`core/Pod`, `apps/Deployment`, ...) as CRDs

    This is required since the new generic control plane scheme doesn't contain those resources any more.

    This is provided by:
    
    - hacks (`HACK` commits) that:
      
      1. <a id="A-2-1"></a> allow the go-restful server to be bypassed for those resources, and route them to the CRD handler
      2. <a id="A-2-2"></a> allow the CRD handler, and opanapi publisher, to support resources of the `core` group
      3. <a id="A-2-3"></a> convert the `protobuf` requests sent to those resources resources to requests in the `application/json` content type, before letting the CRD handler serve them
      4. <a id="A-2-4"></a> replace the table converter of CRDs that bring back those resources, with the default table converter of the related built-in resource

    - a new feature, or potential kube fix (`KUBEFIX` commit), that:
      
      5. <a id="A-2-5"></a> introduces the support of strategic merge patch for CRDs.
          This support uses the OpenAPI v3 schema of the CRD to drive the SMP execution, but only adds a minimal implementation and doesn't fully support OpenAPI schemas that don't have expected `patchStrategy` and `patchMergeKey` annotations.
          In order to avoid changing the behavior of existing client tools, the support is only added for those K8S built-in resources 

## B. Logical clusters

__Investigation document:__ [logical-clusters.md](https://github.com/kcp-dev/kcp/blob/main/docs/investigations/logical-clusters.md)

1.  Logical clusters represented as a prefix in etcd

    It is mainly provided by hacks (`HACK` commits) that:
    
    1. <a id="B-1-1"></a> allow intercepting the api server handler chain to set the expected logical cluster context value from either a given suffix in the request APIServer base URL, or a given header in the http request

    2. <a id="B-1-2"></a> change etcd storage layer in order to use the logical cluster as a prefix in the etcd key

    3. <a id="B-1-3"></a> allow wildcard watches that retrieve objects from all the logical clusters

    4. <a id="B-1-4"></a> correctly get or set the `clusterName` metadata field in the storage layer operations based on the etcd key and its new prefix

2.  Support of logical clusters (== tenancy) in the CRD management, OpenAPI and discovery endpoints, and clients used by controllers

    <a id="B-2"></a>It is mainly provided by a hack (`HACK` commit) that adds CRD tenancy by ensuring that logical clusters are taken in account in:
    - CRD-related controllers
    - APIServices-related controllers
    - Discovery + OpenAPI endpoints
    
    In the current Kubernetes design, those 3 areas are highly coupled and intricated, which explains why this commit had to hack the code at various levels:
    - client-go level
    - controllers level,
    - http handlers level.

    While this gives a detailed idea of which code needs to be touched in order to enable CRD tenancy, a clean implementation would first require some refactoring, in order to build the required abstraction layers that would allow decoupling those areas.

# Potential client problems

Although these changes in the K8S codebase were made in order to keep the compatibility with Kuberntes client tools, there might be some problems:

## Incomplete protobuf support for built-in resources

In some contexts, like the `controller-runtime` library used by the Operator SDK, all the resources of the `client-go` scheme are created / updated using the `application/vnd.kubernetes.protobuf` content type.
 
However when these resources are in fact added as CRDs, in the KCP minimal API server scenario, these resources cannot be created / updated since the protobuf (de)serialization is not (and probably cannot be) supported for CRDs.
So for now in this case, the [A.2.3 hack mentioned above](#A-2-3) just converts the `protobuf` request to a `json` one, but this might not cover all the use-cases or corner cases.

The clean solution would probably be the negotiation of serialization type in `client-go`, which we haven't implemented yet, but which would work like this:
When a request for an unsupported serialization is returned, the server should reject it with a 406
and provide a list of supported content types. `client-go` should then examine whether it can satisfy such a request by encoding the object with a different scheme.
This would require a KEP but at least is in keeping with content negotiation on GET / WATCH in Kube

## Incomplete Strategic merge patch support for built-in resources

Client tools like `kubectl` assume that all K8S native resources (== `client-go` schema resources)
support strategic merge patch, and use it by default when updating or patching a resource.

In Kube, currently, strategic merge patch is not supported for CRDs, which would break compatibility with client tools for all the K8S natives resources that are in fact added as CRD in the KCP minimal api server.
The [A-2-5 change mentioned above](#A-2-5) tries to fix this by using the CRD openAPI v3 schema as the source of the required information that will drive the strategic merge patch execution.

While this fixes the problem in most cases, there might still be errors in case the OpenAPI v2 schema for such a resource is missing `x-kubernetes-patch-strategy` and `x-kubernetes-patch-merge-key` annotations when imported from the CRD OpenAPI v3 schema.