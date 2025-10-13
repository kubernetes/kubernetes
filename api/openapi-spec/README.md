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

Annotates an array to further describe its topology. This extension must only be used on lists and may have 3 possible values:

- `atomic`: the list is treated as a single entity, like a scalar. Atomic lists will be entirely replaced when updated. This extension may be used on any type of list (struct, scalar, ...).
- `set`: Sets are lists that must not have multiple items with the same value. Each value must be a scalar, an object with x-kubernetes-map-type `atomic` or an array with x-kubernetes-list-type `atomic`.
- `map`: These lists are like maps in that their elements have a non-index key used to identify them. Order is preserved upon merge. The map tag must only be used on a list with elements of type object. Requires `x-kubernetes-list-map-keys` to specify which fields uniquely identify list elements.

For example:

``` json
"paths": {
    ...
    "/api/v1/namespaces/{namespace}/pods/{name}": {
        ...
        "put": {
            ...
            "schema": {
                "properties": {
                    "spec": {
                        "properties": {
                            "containers": {
                                "items": {
                                    "properties": {
                                        "env": {
                                            "description": "List of environment variables to set in the container.",
                                            "type": "array",
                                            "items": {
                                                ...
                                            },
                                            "x-kubernetes-list-type": "map",
                                            "x-kubernetes-list-map-keys": ["name"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### `x-kubernetes-list-map-keys`

Used in conjunction with `x-kubernetes-list-type: map` to specify one or more key fields that uniquely identify elements in a list. The value is an array of strings, where each string is the name of a field in the list element's schema that serves as a key. When multiple keys are specified, the combination of all key values must be unique across list elements.

For example, a single key:

``` json
"env": {
    "type": "array",
    "items": {
        ...
    },
    "x-kubernetes-list-type": "map",
    "x-kubernetes-list-map-keys": ["name"]
}
```

For example, multiple keys:

``` json
"ports": {
    "type": "array",
    "items": {
        ...
    },
    "x-kubernetes-list-type": "map",
    "x-kubernetes-list-map-keys": ["containerPort", "protocol"]
}
```

### `x-kubernetes-map-type`

Annotates an object to further describe its topology. This extension must only be used when type is object and may have 2 possible values:

- `granular`: These maps are actual maps (key-value pairs) and each field is independent from each other (they can each be manipulated by separate actors). This is the default behaviour for all maps.
- `atomic`: the map is treated as a single entity, like a scalar. Atomic maps will be entirely replaced when updated.

For example:

``` json
"io.k8s.api.core.v1.ConfigMapKeySelector": {
    "properties": {
        "key": {
            "description": "The key to select.",
            "type": "string"
        },
        "name": {
            "description": "Name of the referent.",
            "type": "string"
        },
        "optional": {
            "description": "Specify whether the ConfigMap or its key must be defined",
            "type": "boolean"
        }
    },
    "required": ["key"],
    "type": "object",
    "x-kubernetes-map-type": "atomic"
}
```

### `x-kubernetes-unions`

Defines discriminated unions (mutually exclusive fields). Can include:

- `discriminator`: String containing the field name used to differentiate union variants
- `fields-to-discriminateBy`: Map of field names to discriminator values; for each entry, the key is the name of one of the mutually exclusive fields, and the value is the string that must be set in the `discriminator` field when that mutually exclusive field is used.

For example:

``` json
"io.k8s.api.flowcontrol.v1.PriorityLevelConfigurationSpec": {
    "properties": {
        "type": {
            "description": "type indicates whether this priority level is 'Exempt' or 'Limited'",
            "type": "string"
        },
        "exempt": {
            "description": "exempt specifies how requests are handled when type is 'Exempt'",
            ...
        },
        "limited": {
            "description": "limited specifies how requests are handled when type is 'Limited'",
            ...
        }
    },
    "required": ["type"],
    "type": "object",
    "x-kubernetes-unions": [
        {
            "discriminator": "type",
            "fields-to-discriminateBy": {
                "exempt": "Exempt",
                "limited": "Limited"
            }
        }
    ]
}
```
