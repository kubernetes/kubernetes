# validation-gen with a custom tag prefix

This directory is a complete, working example of a project that builds its
own validation generator on top of validation-gen. The generator:

* recognizes every standard validation tag under the project's own prefix,
  `+xyz:` instead of `+k8s:` (`+xyz:validation-gen=*`, `+xyz:required`,
  `+xyz:minimum=1`, `+xyz:eachVal=...`, and so on), and
* adds a tag of its own, `+xyz:startsWith="..."`, backed by a validation
  function that lives in the project rather than in `k8s.io/apimachinery`.

The layout mirrors what a third-party Go module would contain:

| Path | Purpose |
| --- | --- |
| `main.go` | The generator binary. It is `k8s.io/code-generator/cmd/validation-gen/main.go` with the tag prefix set and the project's tags imported. |
| `tags/` | The project's `validators.TagValidator` implementations. Importing the package registers them. |
| `validate/` | The runtime validation functions that generated code calls for the project's tags. |
| `output_tests/` | Generated-output tests for the project's tags (`startswith`) and for the standard tags under the new prefix (`builtins`). |

## Building the generator

A generator is the standard `main.go` with two changes: set
`args.TagPrefix` before parsing flags (so it is the default for
`--tag-prefix`), and import the package that registers the project's tags.

```go
import (
	"k8s.io/code-generator/cmd/validation-gen/args"
	"k8s.io/code-generator/cmd/validation-gen/generators"

	_ "example.com/xyz/hack/validation-gen/tags" // registers +xyz:* tags
)

func main() {
	args := args.New()
	args.TagPrefix = "xyz:"
	// ... flag parsing, args.Validate(), gengo.Execute(generators.GetTargets, ...)
}
```

The prefix must be empty or one or more `:`-terminated segments. It applies to
every tag validation-gen reads: the package-level tags that configure
generation (`+xyz:validation-gen`, `+xyz:validation-gen-scheme-registry`,
`+xyz:validation-gen-nolint`, ...), the type- and field-level validation
tags, and the tags that lint rules look for (`+xyz:optional`, `+xyz:alpha`,
...). The `+default` tag is not prefixed; it is shared with defaulter-gen.

An empty prefix is allowed but claims unprefixed tags such as `+optional` and
`+required`, which other tools also read, so choose it deliberately.

A project that only wants a different prefix, and no new tags, does not need
its own binary: `validation-gen --tag-prefix xyz:` does the same thing.

## Defining a tag

A tag is a `validators.TagValidator`, registered from an `init()` function
with `validators.RegisterTagValidator`. `TagName()` returns the name relative
to the prefix (`"startsWith"`, not `"xyz:startsWith"`); the registry
qualifies it. See `tags/startswith.go`.

`GetValidations` returns the function calls to emit. Each is a
`validators.FunctionGen` naming a Go function that has the standard
validation signature, plus any extra arguments taken from the tag:

```go
func StartsWith[T ~string](ctx context.Context, op operation.Operation,
	fldPath *field.Path, value, oldValue *T, prefix string) field.ErrorList
```

The function can live anywhere the generated code can import; here it is in
`validate/`. The standard tags keep calling `k8s.io/apimachinery/pkg/api/validate`.

Tags that need to name other tags, in error messages or when inspecting a
nested tag value, prepend `Config.TagPrefix` (received in `Init`) to the
relative name.

## Testing a tag

Each package under `output_tests/` is a self-contained fixture:

* `doc.go` declares types with the tags under test, activates the generator
  with `+xyz:validation-gen=...`, and registers with the test scheme via
  `+xyz:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme`.
* `zz_generated.validations.go` is the generator's output for that package.
* `doc_test.go` validates values through `localSchemeBuilder.Test(t)` and
  asserts the resulting `field.ErrorList`.

`output_tests/generate.go` carries the `go:generate` directive that runs the
generator over the tree; run `go generate .` in that directory after changing
a tag or a fixture, then `go test ./...`. The `testscheme` package is importable
by any module, so third-party projects can use the same pattern.
