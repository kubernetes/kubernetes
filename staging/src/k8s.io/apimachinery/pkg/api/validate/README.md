# API validation

This package holds functions which validate fields and types in the Kubernetes
API. It may be useful beyond API validation, but this is the primary goal.

Most of the public functions here have signatures which adhere to the following
pattern, which is assumed by automation and code-generation:

```
import (
        "context"
        "k8s.io/apimachinery/pkg/api/operation"
        "k8s.io/apimachinery/pkg/util/validation/field"
)

func <Name>(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue <ValueType>, <OtherArgs...>) field.ErrorList
```

The name of validator functions should consider that callers will generally be
spelling out the package name and the function name, and so should aim for
legibility.  E.g. `validate.Concept()`.

The `ctx` argument is Go's usual Context.

The `opCtx` argument provides information about the API operation in question.

The `fldPath` argument indicates the path to the field in question, to be used
in errors.

The `value` and `oldValue` arguments are the thing(s) being validated.  For
CREATE operations (`opCtx.Operation == operation.Create`), the `oldValue`
argument will be nil.  Many validators functions only look at the current value
(`value`) and disregard `oldValue`.

The `value` and `oldValue` arguments are always nilable - pointers to primitive
types, slices of any type, or maps of any type.  Validator functions should
avoid dereferencing nil. Callers are expected to not pass a nil `value` unless the
API field itself was nilable. `oldValue` is always nil for CREATE operations and 
is also nil for UPDATE operations if the `value` is not correlated with an `oldValue`.

Simple content-validators may have no `<OtherArgs>`, but validator functions
may take additional arguments.  Some validator functions will be built as
generics, e.g. to allow any integer type or to handle arbitrary slices.

Examples:

```
// NonEmpty validates that a string is not empty.
func NonEmpty(ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *string) field.ErrorList

// Even validates that a slice has an even number of items.
func Even[T any](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ []T) field.ErrorList

// KeysMaxLen validates that all of the string keys in a map are under the
// specified length.
func KeysMaxLen[T any](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ map[string]T, maxLen int) field.ErrorList
```

Validator functions always return an `ErrorList` where each item is a distinct
validation failure and a zero-length return value (not just nil) indicates
success.

Good validation failure messages follow the Kubernetes API conventions, for
example using "must" instead of "should".
