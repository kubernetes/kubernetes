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

Definitions may have `x-kubernetes-list-type` to annotate arrays and further describe their topology.
This extension is used to define how lists should be treated during updates and merges.

The extension can have one of three values:

- `atomic`: The list is treated as a single entity. Any update replaces the entire list.
- `set`: Each element in the list must be unique. Elements must be scalars, objects with `x-kubernetes-map-type: atomic`, or arrays with `x-kubernetes-list-type: atomic`.
- `map`: Elements are identified by a key field (specified by `x-kubernetes-list-map-keys`), allowing individual elements to be identified and merged while preserving order.

For example:

``` json
"properties": {
    "ports": {
        "type": "array",
        "x-kubernetes-list-type": "atomic",
        "items": {
            "type": "object"
        }
    }
}
```

### `x-kubernetes-list-map-keys`

Definitions may have `x-kubernetes-list-map-keys` when `x-kubernetes-list-type` is set to `map`.
This extension specifies the field(s) that uniquely identify elements in the list, enabling strategic merge patch behavior for lists of objects.

The keys must be:
- Scalar-typed fields within the list element structure
- Either required fields or fields with default values

For example:

``` json
"properties": {
    "containers": {
        "type": "array",
        "x-kubernetes-list-type": "map",
        "x-kubernetes-list-map-keys": ["name"],
        "items": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string"
                }
            }
        }
    }
}
```

### `x-kubernetes-unions`

Definitions may have `x-kubernetes-unions` to define discriminated union types.
This extension ensures that exactly one field among several options is configured, using a discriminant field to explicitly declare which option is active.

The discriminant field should be:
- A required string (or string alias) type
- Named to clearly indicate which union member is selected
- Typically in PascalCase, corresponding to the camelCase JSON field name of the union member

For example:

``` json
"properties": {
    "type": {
        "type": "string",
        "description": "Discriminant field indicating which volume type is configured"
    },
    "hostPath": {
        "type": "object",
        "description": "HostPath volume configuration"
    },
    "emptyDir": {
        "type": "object",
        "description": "EmptyDir volume configuration"
    }
},
"x-kubernetes-unions": [
    {
        "discriminator": "type",
        "fields-to-discriminateBy": {
            "hostPath": "HostPath",
            "emptyDir": "EmptyDir"
        }
    }
]
```
