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

Indicates how lists are interpreted and merged. Valid values include:

atomic – the entire list is treated as a single scalar value

map – the list is treated as a map, using x-kubernetes-list-map-keys

set – items are treated as a set

Example:

type: object
properties:
  ports:
    type: array
    x-kubernetes-list-type: map
    x-kubernetes-list-map-keys:
      - name

### `x-kubernetes-list-map-keys`

Used together with x-kubernetes-list-type: map.
Specifies which fields of list items should be used as the map key (i.e., the unique identifier for merging).

Example:

type: object
properties:
  ports:
    type: array
    x-kubernetes-list-type: map
    x-kubernetes-list-map-keys:
      - name

### `x-kubernetes-unions`

Represents union (one-of) semantics in OpenAPI schemas, allowing only one field in the union to be set at a time.
Used for modeling discriminated union types.

Example:

type: object
x-kubernetes-unions:
  - discriminator: type
    fields:
      foo: FooType
      bar: BarType
properties:
  type:
    type: string
    enum:
      - Foo
      - Bar
  foo:
    $ref: "#/definitions/FooType"
  bar:
    $ref: "#/definitions/BarType"