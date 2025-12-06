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

Some definitions may have `x-kubernetes-list-type`.  
This extension describes how Kubernetes treats list (array) fields when merging objects.

Valid values are:

- `"atomic"` – the entire list is treated as a single value; updates replace the whole list.
- `"set"` – the list behaves like a set; items are merged based on equality.
- `"map"` – the list behaves like a map; uniqueness is determined by keys defined in `x-kubernetes-list-map-keys`.

For example:

```json
{
  "x-kubernetes-list-type": "map",
  "x-kubernetes-list-map-keys": ["name"]
}
```

### `x-kubernetes-list-map-keys`

Some definitions may have `x-kubernetes-list-map-keys`.  
This extension identifies which field(s) inside list items act as keys when `x-kubernetes-list-type` is `"map"`.

Each key must refer to a field within the list element.  
Items with matching key values are merged; items with different keys are appended.

For example:

```json
{
  "x-kubernetes-list-type": "map",
  "x-kubernetes-list-map-keys": ["name"]
}
```

### `x-kubernetes-unions`

Some definitions may have `x-kubernetes-unions`.  
This extension describes union-like structures where only one of several fields may be set.  
A discriminator field determines which field is valid.

For example:

```yaml
x-kubernetes-unions:
  - discriminator: type
    fields-to-discriminateBy:
      httpGet: HTTPGet
      tcpSocket: TCPSocket
```

In this example, when `type` is `"HTTPGet"`, only the `httpGet` field may appear;  
when `type` is `"TCPSocket"`, only the `tcpSocket` field may appear.

