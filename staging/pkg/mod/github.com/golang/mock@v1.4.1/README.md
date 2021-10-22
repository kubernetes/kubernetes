gomock [![Build Status][travis-ci-badge]][travis-ci] [![GoDoc][godoc-badge]][godoc]
======

GoMock is a mocking framework for the [Go programming language][golang]. It
integrates well with Go's built-in `testing` package, but can be used in other
contexts too.

Installation
------------

Once you have [installed Go][golang-install], install the `mockgen` tool.

To get the latest released version use:

```bash
GO111MODULE=on go get github.com/golang/mock/mockgen@latest
```

If you use `mockgen` in your CI pipeline, it may be more appropriate to fixate
on a specific mockgen version.


Documentation
-------------

After installing, you can use `go doc` to get documentation:

```bash
go doc github.com/golang/mock/gomock
```

Alternatively, there is an online reference for the package hosted on GoPkgDoc
[here][gomock-ref].

Running mockgen
---------------

`mockgen` has two modes of operation: source and reflect.
Source mode generates mock interfaces from a source file.
It is enabled by using the -source flag. Other flags that
may be useful in this mode are -imports and -aux_files.

Example:

```bash
mockgen -source=foo.go [other options]
```

Reflect mode generates mock interfaces by building a program
that uses reflection to understand interfaces. It is enabled
by passing two non-flag arguments: an import path, and a
comma-separated list of symbols.

You can use "." to refer to the current path's package.

Example:

```bash
mockgen database/sql/driver Conn,Driver

# Convenient for `go:generate`.
mockgen . Conn,Driver
```

The `mockgen` command is used to generate source code for a mock
class given a Go source file containing interfaces to be mocked.
It supports the following flags:

* `-source`: A file containing interfaces to be mocked.

* `-destination`: A file to which to write the resulting source code. If you
    don't set this, the code is printed to standard output.

* `-package`: The package to use for the resulting mock class
    source code. If you don't set this, the package name is `mock_` concatenated
    with the package of the input file.

* `-imports`: A list of explicit imports that should be used in the resulting
    source code, specified as a comma-separated list of elements of the form
    `foo=bar/baz`, where `bar/baz` is the package being imported and `foo` is
    the identifier to use for the package in the generated source code.

* `-aux_files`: A list of additional files that should be consulted to
    resolve e.g. embedded interfaces defined in a different file. This is
    specified as a comma-separated list of elements of the form
    `foo=bar/baz.go`, where `bar/baz.go` is the source file and `foo` is the
    package name of that file used by the -source file.

* `-build_flags`: (reflect mode only) Flags passed verbatim to `go build`.

* `-mock_names`: A list of custom names for generated mocks. This is specified
    as a comma-separated list of elements of the form
    `Repository=MockSensorRepository,Endpoint=MockSensorEndpoint`, where
    `Repository` is the interface name and `MockSensorRepository` is the desired
    mock name (mock factory method and mock recorder will be named after the mock).
    If one of the interfaces has no custom name specified, then default naming
    convention will be used.
    
* `-self_package`: The full package import path for the generated code. The purpose 
    of this flag is to prevent import cycles in the generated code by trying to include 
    its own package. This can happen if the mock's package is set to one of its 
    inputs (usually the main one) and the output is stdio so mockgen cannot detect the 
    final output package. Setting this flag will then tell mockgen which import to exclude.

* `-copyright_file`: Copyright file used to add copyright header to the resulting source code.

For an example of the use of `mockgen`, see the `sample/` directory. In simple
cases, you will need only the `-source` flag.

Building Mocks
--------------

```go
type Foo interface {
  Bar(x int) int
}

func SUT(f Foo) {
 // ...
}

```

```go
func TestFoo(t *testing.T) {
  ctrl := gomock.NewController(t)

  // Assert that Bar() is invoked.
  defer ctrl.Finish()

  m := NewMockFoo(ctrl)

  // Asserts that the first and only call to Bar() is passed 99.
  // Anything else will fail.
  m.
    EXPECT().
    Bar(gomock.Eq(99)).
    Return(101)

  SUT(m)
}
```

Building Stubs
--------------

```go
type Foo interface {
  Bar(x int) int
}

func SUT(f Foo) {
 // ...
}

```

```go
func TestFoo(t *testing.T) {
  ctrl := gomock.NewController(t)
  defer ctrl.Finish()

  m := NewMockFoo(ctrl)

  // Does not make any assertions. Returns 101 when Bar is invoked with 99.
  m.
    EXPECT().
    Bar(gomock.Eq(99)).
    Return(101).
    AnyTimes()

  // Does not make any assertions. Returns 103 when Bar is invoked with 101.
  m.
    EXPECT().
    Bar(gomock.Eq(101)).
    Return(103).
    AnyTimes()

  SUT(m)
}
```

### Modifying Failure Messages

When a matcher reports a failure, it prints the received (`Got`) vs the
expected (`Want`) value.

```
Got: [3]
Want: is equal to 2
Expected call at user_test.go:33 doesn't match the argument at index 1.
Got: [0 1 1 2 3]
Want: is equal to 1
```

##### Modifying `Want`

The `Want` value comes from the matcher's `String()` method. If the matcher's
default output doesn't meet your needs, then it can be modified as follows:

```go
gomock.WantFormatter(
  gomock.StringerFunc(func() string { return "is equal to fifteen" }),
  gomock.Eq(15),
)
```

This modifies the `gomock.Eq(15)` matcher's output for `Want:` from `is equal
to 15` to `is equal to fifteen`.

##### Modifying `Got`

The `Got` value comes from the object's `String()` method if it is available.
In some cases the output of an object is difficult to read (e.g., `[]byte`) and
it would be helpful for the test to print it differently. The following
modifies how the `Got` value is formatted:

```go
gomock.GotFormatterAdapter(
  gomock.GotFormatterFunc(func(i interface{}) string {
    // Leading 0s
    return fmt.Sprintf("%02d", i)
  }),
  gomock.Eq(15),
)
```

If the received value is `3`, then it will be printed as `03`.

[golang]:          http://golang.org/
[golang-install]:  http://golang.org/doc/install.html#releases
[gomock-ref]:      http://godoc.org/github.com/golang/mock/gomock
[travis-ci-badge]: https://travis-ci.org/golang/mock.svg?branch=master
[travis-ci]:       https://travis-ci.org/golang/mock
[godoc-badge]:     https://godoc.org/github.com/golang/mock/gomock?status.svg
[godoc]:           https://godoc.org/github.com/golang/mock/gomock
