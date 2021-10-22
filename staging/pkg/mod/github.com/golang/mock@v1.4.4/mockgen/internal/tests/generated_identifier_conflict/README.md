# Generated Identifier Conflict

The generated mock methods use some hardcoded variable/receiver names that can
have conflicts with the argument names that are defined by the code for which
the mock is generated when using the source generation method.

Example:

```go
type Example interface {
    Method(_m, _mr, m, mr int)
}
```

```go
// Method mocks base method
func (_m *MockExample) Method(_m int, _mr int, m int, mr int) {
    _m.ctrl.Call(_m, "Method", _m, _mr, m, mr)
}
```

In the above example one of the interface method parameters is called `_m`
but unfortunately the generated receiver name is also called `_m` so the
mock code won't compile.

The generator has to make sure that generated identifiers (e.g.: the receiver
names) are always different from the arg names that might come from external
sources.
