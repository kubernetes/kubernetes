# inject
--
    import "github.com/codegangsta/inject"

Package inject provides utilities for mapping and injecting dependencies in
various ways.

Language Translations:
* [简体中文](translations/README_zh_cn.md)

## Usage

#### func  InterfaceOf

```go
func InterfaceOf(value interface{}) reflect.Type
```
InterfaceOf dereferences a pointer to an Interface type. It panics if value is
not an pointer to an interface.

#### type Applicator

```go
type Applicator interface {
	// Maps dependencies in the Type map to each field in the struct
	// that is tagged with 'inject'. Returns an error if the injection
	// fails.
	Apply(interface{}) error
}
```

Applicator represents an interface for mapping dependencies to a struct.

#### type Injector

```go
type Injector interface {
	Applicator
	Invoker
	TypeMapper
	// SetParent sets the parent of the injector. If the injector cannot find a
	// dependency in its Type map it will check its parent before returning an
	// error.
	SetParent(Injector)
}
```

Injector represents an interface for mapping and injecting dependencies into
structs and function arguments.

#### func  New

```go
func New() Injector
```
New returns a new Injector.

#### type Invoker

```go
type Invoker interface {
	// Invoke attempts to call the interface{} provided as a function,
	// providing dependencies for function arguments based on Type. Returns
	// a slice of reflect.Value representing the returned values of the function.
	// Returns an error if the injection fails.
	Invoke(interface{}) ([]reflect.Value, error)
}
```

Invoker represents an interface for calling functions via reflection.

#### type TypeMapper

```go
type TypeMapper interface {
	// Maps the interface{} value based on its immediate type from reflect.TypeOf.
	Map(interface{}) TypeMapper
	// Maps the interface{} value based on the pointer of an Interface provided.
	// This is really only useful for mapping a value as an interface, as interfaces
	// cannot at this time be referenced directly without a pointer.
	MapTo(interface{}, interface{}) TypeMapper
	// Provides a possibility to directly insert a mapping based on type and value.
	// This makes it possible to directly map type arguments not possible to instantiate
	// with reflect like unidirectional channels.
	Set(reflect.Type, reflect.Value) TypeMapper
	// Returns the Value that is mapped to the current type. Returns a zeroed Value if
	// the Type has not been mapped.
	Get(reflect.Type) reflect.Value
}
```

TypeMapper represents an interface for mapping interface{} values based on type.
