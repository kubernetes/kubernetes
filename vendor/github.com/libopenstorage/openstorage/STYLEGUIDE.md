# Style Guide

This is the official openstorage style guide for golang code. This is in addition to the offical style guide at https://github.com/golang/go/wiki/CodeReviewComments.

This is just a rough outline for now, we will formalize this as we go.

IF YOU CODE ON OPENSTORAGE, YOU ARE EXPECTED TO KNOW THIS. Just take the 20 minutes and read through the issues, we will buy you a coffee, maybe.

### Relevant Issues

* https://github.com/libopenstorage/openstorage/issues/100
* https://github.com/libopenstorage/openstorage/issues/88
* https://github.com/libopenstorage/openstorage/issues/97
* https://github.com/libopenstorage/openstorage/issues/87
* https://github.com/libopenstorage/openstorage/issues/92
* https://github.com/libopenstorage/openstorage/issues/89
* https://github.com/libopenstorage/openstorage/issues/76
* https://github.com/libopenstorage/openstorage/issues/71
* https://github.com/libopenstorage/openstorage/issues/96

### Items

* Use [dlog](https://go.pedge.io/dlog) for logging.

* File order:

```go
package pkg

const (
  ...
)

var (
  ...
)

// but init should generally be not used
func init() {
}

// public struct

// public struct functions

// private struct functions

// private functions but only if they just apply to this struct, otherwise in a common file
```

* All new code must pass `go vet`, `errcheck`, `golint`. For `errcheck`, this means no more unchecked errors. Use of `_ = someFnThatReturnsError()` will be audited soon, and in general should not be used anymore. `golint` means all public types need comments to pass, which means both (A) we will have better code documentation and (B) we will think more about what should actually be public. `errcheck` and `golint` are great detterrents.

* All packages have a file named `name_of_package.go`, ie in `api/server`, we have `server.go`.

* Packages are named after their directory, ie `api/` is `package api`.

* **ALL PUBLIC TYPES GO IN `name_of_package.go`.** Every other file is just a helper file that implements the types.

* Variable names should reflect the type, ie an instance of  `Runner` should be `runner`. This is in contrast to golang's official recommendation, but has been found to make code more readable. Heh. So this means `api.Volume` is not `v, vol, whatever`, it's `volume`, a `request` is not `req, createReq, r`, it's `request`. Only exception is the receiver argument on a function, ie `func (s *server) Foo(...) { ... }`.

* Structs without a corresponding interface are data holders. Structs with functions attached have a public interface wrapper, and then the struct becomes private. Example:

```go
// foo.go
package foo

type Runner interace {
  Run(one string, i int) error
}

func NewRunner(something bar.Something) Runner {
  return newRunner(something)
}

// runner.go
package foo

type runner struct{
  something bar.Something
}

func newRunner(something bar.Something) *runner {
  return &runner{something}
}

func (r *runner) Run(one string, i int) error {
  r.hello(i)
  return r.something.Bar(one, i+1)
}

func (r *runner) hello(i int) {
  return r.something.Hello(i)
}
```
              
* Most structs that have functions attached have a separate file with the private struct definition, private constructor, and public functions, then private functions. The runner struct above is an example.

* Use struct pointers in general instead of structs. It's a debate, but not for now.

* Function parameters/struct initialization/function calls etc are either on one line or each parameter/field has a new line for it. Example:

```go
// yes
function oneLine(a string, b string, c string) {
}

//yes
function multiLine(
  a string,
  b string,
  c string,
) {
}

//no
function multiLineNo(
  a string,
  b string,
  c string) {
}

// no
function multiLineNo2(a string,
  b string, c string) {
}
```

* **NO CALLING `os.Exit(...)` OR `panic(...)` IN LIBRARY CODE.** Ie nowhere but a main package.

* New introductions of global variables and init functions have to be vetted extensively by project owners (and existing ones should be deleted as much as we can).

* No reliance on freeform string matching for errors.

* No typing dynamic value primitives. Example:

```go
// no
type VolumeID string

type Volume struct {
  VolumeID VolumeID
  ...
}

//yes
type Volume struct {
  VolumeID string  
}
```

* Static value primitives (also known as enums) are not strings. Most new ones should be generated with protobuf, see [api/api.proto](api/api.proto) for examples.

* Remove most uses of private variables in public structs.

* Remove most extra variable definitions that are not needed or turn into constants (https://github.com/libopenstorage/openstorage/blob/8d07329468ef709838e443dc17b1eecf2c7cf77d/api/server/volume.go#L76).

* Reduce adding of String() methods on most objects (let the generic `%+v` take care of it).

* Use less newlines within methods.

* Single errors are scoped within an if statement:

```go
// no
err := foo()
if err != nil {
  return nil, err
}

// yes
if err := foo(); err != nil {
  return nil, err
}

// yes, if ignoring return value
// if _, err := bar(); err != nil {
  return nil, err
}
```
    
* Empty structs:

```go
// no
type EmptyStruct {
}
// yes
type EmptyStruct {}
```

* Blank imports should have explanation (https://github.com/libopenstorage/openstorage/blob/8d07329468ef709838e443dc17b1eecf2c7cf77d/volume/enumerator.go#L6).

* No code checked in that has warnings https://golang.org/cmd/cgo/.

* Do not check in if `make docker-test` does not pass.
