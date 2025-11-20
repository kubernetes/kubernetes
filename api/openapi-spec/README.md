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

This extension specifies the type of list. It can be one of:
- `atomic`: the list is treated as a single entity, like a scalar. Atomic lists cannot be merged. The entire list is replaced when updated.
- `set`: the list is treated as a set where order does not matter. The API server will merge lists based on their content.
- `map`: the list is treated as a map where each element has a key used for merging. Must be used with `x-kubernetes-list-map-keys`.

For example:

```json
"properties": {
  "items": {
    "type": "array",
    "x-kubernetes-list-type": "set",
    "items": {
      "type": "string"
    }
  }
}
```

### `x-kubernetes-list-map-keys`

This extension specifies the map keys when `x-kubernetes-list-type` is `map`. It is an array of strings that represent the fields that uniquely identify list items.

For example:

```json
"properties": {
  "items": {
    "type": "array",
    "x-kubernetes-list-type": "map",
    "x-kubernetes-list-map-keys": ["name", "port"],
    "items": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "port": {"type": "integer"},
        "protocol": {"type": "string"}
      }
    }
  }
}
```

### `x-kubernetes-unions`

This extension is used to define discriminated unions in the API. A union is a structure where only one field can be set at a time. It's typically used with a discriminator field that indicates which field is currently set.

For example:

```json
"x-kubernetes-unions": [
  {
    "discriminator": "type",
    "fields-to-discriminateBy": {
      "httpGet": "HTTPGet",
      "tcpSocket": "TCPSocket"
    }
  }
]
```

In this example, either `httpGet` or `tcpSocket` can be set, but not both. The `type` field acts as a discriminator to indicate which field is currently set.
