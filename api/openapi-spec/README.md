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

### `x-kubernetes-list-map-keys`

Operations and Definitions may have `x-kubernetes-list-maps-keys` if they
are associated with a [kubernetes resource](https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources). `x-kubernetes-list-type` = `map` specifies field names inside each list element to serve as unique keys for the list-as-map.

**For example:**

```json
{
  "type": "object",
  "properties": {
    "servers": {
      "type": "array",
      "x-kubernetes-list-type": "map",
      "x-kubernetes-list-map-keys": ["name"],
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "address": { "type": "string" }
        },
        "required": ["name"]
      }
    }
  }
}
```

### `x-kubernetes-patch-strategy` and `x-kubernetes-patch-merge-key`

Some of the definitions may have these extensions. For more information about PatchStrategy and PatchMergeKey see
[strategic-merge-patch](https://git.k8s.io/community/contributors/devel/sig-api-machinery/strategic-merge-patch.md).

### `x-kubernetes-list-type`

Some definitions may have `x-kubernetes-list-type`.  
This extension specifies how lists are merged when Kubernetes objects are combined (for example, during updates or patches).  
Possible values are:

- `"atomic"` – treat the entire list as one unit; replacing the list replaces all elements.  
- `"set"` – treat the list as a set; merge items based on equality.  
- `"map"` – treat the list as a map; keys come from the field defined in `x-kubernetes-list-map-keys`.  

For example:

```json
{
  "x-kubernetes-list-type": "map",
  "x-kubernetes-list-map-keys": ["name"]
}
```

See [Kubernetes API conventions (lists and maps)](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md) for more.

### `x-kubernetes-list-map-keys`

This extension sets which field(s) identify elements uniquely for x-kubernetes-list-type: map.
It allows merging list entries based on those key fields.

For example:

```json
{
  "x-kubernetes-list-type": "map",
  "x-kubernetes-list-map-keys": ["name"]
}
```

Here, "name" acts as the key for each map entry.
See [API conventions – merge strategy](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#merge-strategy) for details.


### `x-kubernetes-map-type`

Some definitions may have x-kubernetes-map-type.
This extension describes how maps should merge:
 • "atomic" – replace entire map
 • "granular" – merge map entries individually

For example:

```json
{
  "x-kubernetes-map-type": "granular"
}
```
See Kubernetes [API conventions (maps section)](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#maps) for details.


### `x-kubernetes-unions`

Some definitions may have x-kubernetes-unions.
This extension describes mutually exclusive fields (union-like behavior).
Only one field in that union can be set.

For example:

```json
{
  "x-kubernetes-unions": [
    {
      "discriminator": "type",
      "fields": ["intValue", "stringValue"]
    }
  ]
}
```

This ensures only one of "intValue" or "stringValue" is specified based on the type discriminator.
See Kubernetes [API conventions (union types)](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#union-types) for more.
