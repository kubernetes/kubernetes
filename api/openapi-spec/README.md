# Kubernetes's OpenAPI Specification

This folder contains an [OpenAPI specification](https://github.com/OAI/OpenAPI-Specification) for Kubernetes API.

## Vendor Extensions

Kubernetes extends OpenAPI using these extensions. Note the version that
extensions have been added.

### `x-kubernetes-group-version-kind`

Operations and Definitions may have `x-kubernetes-group-version-kind` if they
are associated with a [kubernetes resource](https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources).


For example:

``` json
"paths": {
    ...
    "/api/v1/namespaces/{namespace}/pods/{name}": {
        ...
        "get": {
        ...
            "x-kubernetes-group-version-kind": {
            "group": "",
            "version": "v1",
            "kind": "Pod"
            }
        }
    }
}
```

### `x-kubernetes-action`

Operations and Definitions may have `x-kubernetes-action` if they
are associated with a [kubernetes resource](https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources).
Action can be one of `get`, `list`, `put`, `patch`, `post`, `delete`, `deletecollection`, `watch`, `watchlist`, `proxy`, or `connect`.


For example:

``` json
"paths": {
    ...
    "/api/v1/namespaces/{namespace}/pods/{name}": {
        ...
        "get": {
        ...
            "x-kubernetes-action": "list"
        }
    }
}
```

### `x-kubernetes-patch-strategy` and `x-kubernetes-patch-merge-key`

Some of the definitions may have these extensions. For more information about PatchStrategy and PatchMergeKey see
[strategic-merge-patch](https://git.k8s.io/community/contributors/devel/sig-api-machinery/strategic-merge-patch.md).

### `x-kubernetes-list-type`
Annotates an array to further describe its topology.  This extension must only be used on lists, and has three possible values:
- **atomic**: the entire list is treated as a single unit—patch/merge operations replace it wholesale.
- **set**: the list is managed as a set; merge operations perform a union based on element identity.
- **map**: the list is managed as a map; merge operations merge items by key (see `x-kubernetes-map-key`).:contentReference[oaicite:0]{index=0}

### `x-kubernetes-map-key`
When `x-kubernetes-list-type: map` is set, this extension specifies the name of the field in each list element to use as the key in the map.  
The named field must be a scalar property that is either required or has a default value to guarantee uniqueness. :contentReference[oaicite:1]{index=1}

### `x-kubernetes-unions`
Describes a union (one-of) grouping of sub-schemas within an object.  Its value is an object with:
- `discriminator`: the property name whose value indicates which sub-schema applies  
- `properties`: a map from discriminator values to the corresponding schema `$ref`  
Use this to model “tagged unions” in CRD schemas. :contentReference[oaicite:2]{index=2}
