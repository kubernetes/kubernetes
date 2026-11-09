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

Operations and Definitions may have `x-kubernetes-list-map-keys` if they
are associated with a [kubernetes resource](https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources).
This extension MUST only be used on arrays whose `x-kubernetes-list-type` is `map`.
It specifies the scalar fields within each list element that together serve as the
unique key for the list-as-map merge behaviour.

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

### `x-kubernetes-list-type`

Annotates an array to describe its topology for server-side apply and
strategic-merge-patch. The extension must only be used on arrays and may
take one of three values:

- `atomic` – the list is treated as a single opaque entity (like a scalar).
  The entire list is replaced on update. Applicable to any element type.
- `set` – the list behaves as a set; duplicate values are forbidden. Each
  value must be a scalar, an `atomic` object (`x-kubernetes-map-type: atomic`),
  or an `atomic` array.
- `map` – the list behaves as a map keyed by the fields named in
  `x-kubernetes-list-map-keys`. Order is preserved on merge. Elements must
  be objects.

**For example:**

```json
{
  "type": "array",
  "x-kubernetes-list-type": "set",
  "items": { "type": "string" }
}
```

### `x-kubernetes-map-type`

Annotates an object (type `object`) to describe its merge topology for
server-side apply. May take one of two values:

- `granular` – (default) each field is managed independently and can be
  updated by separate actors.
- `atomic` – the entire object is treated as a single opaque entity and is
  replaced atomically on update.

**For example:**

```json
{
  "type": "object",
  "x-kubernetes-map-type": "atomic",
  "properties": {
    "host": { "type": "string" },
    "port": { "type": "integer" }
  }
}
```

### `x-kubernetes-preserve-unknown-fields`

When set to `true`, stops the API server pruning step from removing fields
that are not listed in the validation schema. Pruning is re-enabled when
nested `properties` or `additionalProperties` provide their own schema.
The value must be `true` or omitted (`false` is not permitted).

**For example:**

```json
{
  "type": "object",
  "x-kubernetes-preserve-unknown-fields": true
}
```

### `x-kubernetes-embedded-resource`

When set to `true`, declares that the value is an embedded Kubernetes
`runtime.Object` containing `TypeMeta` and `ObjectMeta`. The schema type
must be `object`. `x-kubernetes-preserve-unknown-fields` must also be set
to `true`. Both `TypeMeta` and `ObjectMeta` are validated automatically.

**For example:**

```json
{
  "type": "object",
  "x-kubernetes-embedded-resource": true,
  "x-kubernetes-preserve-unknown-fields": true
}
```

### `x-kubernetes-int-or-string`

When set to `true`, declares that this value may be either an integer or a
string. An empty `type` is then permitted, and `type` as a child of `anyOf`
is allowed when following one of the patterns below:

```json
{ "anyOf": [{ "type": "integer" }, { "type": "string" }] }
```

or

```json
{
  "allOf": [
    { "anyOf": [{ "type": "integer" }, { "type": "string" }] }
  ]
}
```

**For example:**

```json
{
  "x-kubernetes-int-or-string": true,
  "anyOf": [{ "type": "integer" }, { "type": "string" }]
}
```

### `x-kubernetes-validations`

Specifies a list of validation rules written in
[Common Expression Language (CEL)](https://github.com/google/cel-spec).
Rules are evaluated at admission time. The `self` variable is bound to the
value being validated, scoped to the location of the extension in the schema.

Each rule entry supports the following fields:

| Field | Required | Description |
|---|---|---|
| `rule` | yes | CEL expression that must evaluate to `true` for the object to be valid |
| `message` | no | Human-readable failure message (required when `rule` contains line breaks) |
| `messageExpression` | no | CEL expression returning the failure message string |
| `reason` | no | Machine-readable reason: `FieldValueInvalid`, `FieldValueForbidden`, `FieldValueRequired`, or `FieldValueDuplicate` |
| `fieldPath` | no | Relative JSON path to the field that caused the failure |
| `optionalOldSelf` | no | When `true`, makes `oldSelf` a CEL optional so the rule is evaluated even when no old value exists |

**For example:**

```json
{
  "type": "object",
  "properties": {
    "min": { "type": "integer" },
    "max": { "type": "integer" }
  },
  "x-kubernetes-validations": [
    {
      "rule": "self.max >= self.min",
      "message": "max must be greater than or equal to min"
    }
  ]
}
```

### `x-kubernetes-patch-strategy` and `x-kubernetes-patch-merge-key`

Some of the definitions may have these extensions. For more information about PatchStrategy and PatchMergeKey see
[strategic-merge-patch](https://git.k8s.io/community/contributors/devel/sig-api-machinery/strategic-merge-patch.md).
