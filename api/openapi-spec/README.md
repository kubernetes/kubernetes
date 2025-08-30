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


📘 Additional Kubernetes OpenAPI Extensions
Kubernetes uses custom OpenAPI extensions to define advanced behaviors for lists and object fields in CustomResourceDefinitions (CRDs). This section documents the following extensions:

x-kubernetes-list-type

x-kubernetes-list-map-keys

x-kubernetes-unions

These extensions help the API server handle merge, patch, and validation strategies.

🔹 x-kubernetes-list-type
Defines how lists (arrays) behave when resources are merged or patched.

Supported Values
"atomic" (default): Entire list is replaced during updates.

"set": Items are treated as a unique unordered set.

"map": Items are treated as a map, indexed by keys defined via x-kubernetes-list-map-keys.

✅ Example: Set List
```
finalizers:
  type: array
  x-kubernetes-list-type: set
  items:
    type: string
```
Ensures finalizers is treated as a set with no duplicates.

Used with x-kubernetes-list-type: map to identify unique keys for merging individual items in a list.

✅ Example: List Map of Containers
```
containers:
  type: array
  x-kubernetes-list-type: map
  x-kubernetes-list-map-keys:
    - name
  items:
    type: object
    required:
      - name
    properties:
      name:
        type: string
      image:
        type: string
  ```      
Kubernetes treats the array as a map indexed by name. Useful for container specs.

🔹 x-kubernetes-unions
Declares a group of mutually exclusive fields—only one field in the union should be set at a time.

✅ Example: Union of Fields
```
type: object
x-kubernetes-unions:
  - fields:
      - stringValue
      - intValue
properties:
  stringValue:
    type: string
  intValue:
    type: integer
```
Indicates only one of stringValue or intValue should be set. Tools can enforce this during validation.

🚀 Usage Instructions
You can use these extensions in the openAPIV3Schema section of a Kubernetes CustomResourceDefinition. Kubernetes understands them automatically—no plugins or extra tools required.

📄 Sample CRD Using All Three Extensions
```
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: demos.example.com
spec:
  group: example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                finalizers:
                  type: array
                  x-kubernetes-list-type: set
                  items:
                    type: string
                containers:
                  type: array
                  x-kubernetes-list-type: map
                  x-kubernetes-list-map-keys:
                    - name
                  items:
                    type: object
                    required:
                      - name
                    properties:
                      name:
                        type: string
                      image:
                        type: string
                value:
                  type: object
                  x-kubernetes-unions:
                    - fields:
                        - stringValue
                        - intValue
                  properties:
                    stringValue:
                      type: string
                    intValue:
                      type: integer
  scope: Namespaced
  names:
    plural: demos
    singular: demo
    kind: Demo
```
📦 Apply the CRD
```
kubectl apply -f demo-crd.yaml
```
No special flags or configurations are needed. Kubernetes understands these extensions natively when CRDs are registered.

🧠 Notes
These extensions improve patching, merging, and validation behavior for CRDs.

They're especially useful for tools like kubectl apply and server-side apply.

All behavior is handled by the Kubernetes API server—no external setup required.
