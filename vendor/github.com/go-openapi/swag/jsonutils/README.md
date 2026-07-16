 # jsonutils

`jsonutils` exposes a few tools to work with JSON:

- a fast, simple `Concat` to concatenate (not merge) JSON objects and arrays
- `FromDynamicJSON` to convert a data structure into a "dynamic JSON" data structure
- `ReadJSON` and `WriteJSON` behave like `json.Unmarshal` and `json.Marshal`,
   with the ability to use another underlying serialization library through an `Adapter` 
   configured at runtime
- a `JSONMapSlice` structure that may be used to store JSON objects with the order of keys maintained

## Dynamic JSON

We call "dynamic JSON" the go data structure that results from unmarshaling JSON like this:

```go
  var value any
  jsonBytes := `{"a": 1, ... }`
  _ = json.Unmarshal(jsonBytes, &value)
```

In this configuration, the standard library mappings are as follows:

| JSON      | go               |
|-----------|------------------|
| `number`  | `float64`        |
| `string`  | `string`         |
| `boolean` | `bool`           |
| `null`    | `nil`            |
| `object`  | `map[string]any` |
| `array`   | `[]any`          |

## Map slices

When using `JSONMapSlice`, the ordering of keys is ensured by replacing
mappings to `map[string]any` by a `JSONMapSlice` which is an (ordered)
slice of `JSONMapItem`s.

Notice that a similar feature is available for YAML (see [`yamlutils`](../yamlutils)),
with a `YAMLMapSlice` type based on the `JSONMapSlice`.

`JSONMapSlice` is similar to an ordered map, but the keys are not retrieved
in constant time.

Another difference with the the above standard mappings is that numbers don't always map
to a `float64`: if the value is a JSON integer, it unmarshals to `int64`.

See also [some examples](https://pkg.go.dev/github.com/go-openapi/swag/jsonutils#pkg-examples)

## Adapters

`ReadJSON`, `WriteJSON` and `FromDynamicJSON` (which is a combination of the latter two)
are wrappers on top of `json.Unmarshal` and `json.Marshal`.

By default, the adapter merely wraps the standard library.

The adapter may be used to register other JSON serialization libraries,
possibly several ones at the same time.

If the value passed is identified as an "ordered map" (i.e. implements `ifaces.Ordered`
or `ifaces.SetOrdered`, the adapter favors the "ordered" JSON behavior and tries to
find a registered implementation that support ordered keys in objects.

Our standard library implementation supports this.

As of `v0.25.0`, we support through such an adapter the popular `mailru/easyjson`
library, which kicks in when the passed values support the `easyjson.Unmarshaler` 
or `easyjson.Marshaler` interfaces.

In the future, we plan to add more similar libraries that compete on the go JSON
serializers scene.

## Registering an adapter

In package `github.com/go-openapi/swag/easyjson/adapters`, several adapters are available.

Each adapter is an independent go module. Hence you'll pick its dependencies only if you import it.

At this moment we provide:
* `stdlib`: JSON adapter based on the standard library
* `easyjson`: JSON adapter based on the `github.com/mailru/easyjson`

The adapters provide the basic `Marshal` and `Unmarshal` capabilities, plus an implementation
of the `MapSlice` pattern.

You may also build your own adapter based on your specific use-case. An adapter is not required to implement
all capabilities.

Every adapter comes with a `Register` function, possibly with some options, to register the adapter
to a global registry.

For example, to enable `easyjson` to be used in `ReadJSON` and `WriteJSON`, you would write something like:

```go
  import (
	  "github.com/go-openapi/swag/jsonutils/adapters"
	  easyjson "github.com/go-openapi/swag/jsonutils/adapters/easyjson/json"
  )

  func init() {
	  easyjson.Register(adapters.Registry)
  }
```

You may register several adapters. In this case, capability matching is evaluated from the last registered
adapters (LIFO).

## [Benchmarks](./adapters/testintegration/benchmarks/README.md)
