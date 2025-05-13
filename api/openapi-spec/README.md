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

Annotates an array to further describe its topology. This extension must only be used on lists and may have 3 possible values:

- `atomic`: the list is treated as a single entity, like a scalar. Atomic lists will be entirely replaced when updated. This extension may be used on any type of list (struct, scalar, ...).
- `set`: Sets are lists that must not have multiple items with the same value. Each value must be a scalar, an object with x-kubernetes-map-type `atomic` or an array with x-kubernetes-list-type `atomic`.
- `map`: These lists are like maps in that their elements have a non-index key used to identify them. Order is preserved upon merge. The map tag must only be used on a list with elements of type object. By default, arrays are often treated as atomic, but actual behavior depends on the API and patch strategy.


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
                                        "volumeMounts": {
                                            "type": "array",
                                            "x-kubernetes-list-type": "map",
                                            "x-kubernetes-list-map-keys": ["mountPath"]
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

### `x-kubernetes-map-type`

Annotates an object to further describe its topology. This extension must only be used when type is object and may have 2 possible values:

- `granular`: These maps are actual maps (key-value pairs) and each fields are independent from each other (they can each be manipulated by separate actors). This is the default behaviour for all maps.
- `atomic`: the list is treated as a single entity, like a scalar. Atomic maps will be entirely replaced when updated.


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
                        "type": "object",
                        "x-kubernetes-map-type": "atomic"
                    }
                }
            }
        }
    }
}
```

### `x-kubernetes-unions`

Defines discriminated unions (mutually exclusive fields). Requires:

- `discriminator`: Field name used to differentiate union variants
- `fields-to-discriminateBy`: Mapping of discriminator values to required fields


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
                            "securityContext": {
                                "properties": {
                                    "appArmorProfile": {
                                        "type": "object",
                                        "x-kubernetes-unions": [
                                            {
                                                "discriminator": "type",
                                                "fields-to-discriminateBy": {
                                                    "localhostProfile": "LocalhostProfile"
                                                }
                                            }
                                        ]
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
