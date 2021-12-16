# genny - Generics for Go

[![Build Status](https://travis-ci.org/cheekybits/genny.svg?branch=master)](https://travis-ci.org/cheekybits/genny) [![GoDoc](https://godoc.org/github.com/cheekybits/genny/parse?status.png)](http://godoc.org/github.com/cheekybits/genny/parse)

Install:

```
go get github.com/cheekybits/genny
```

=====

(pron. Jenny) by Mat Ryer ([@matryer](https://twitter.com/matryer)) and Tyler Bunnell ([@TylerJBunnell](https://twitter.com/TylerJBunnell)).

Until the Go core team include support for [generics in Go](http://golang.org/doc/faq#generics), `genny` is a code-generation generics solution. It allows you write normal buildable and testable Go code which, when processed by the `genny gen` tool, will replace the generics with specific types.

  * Generic code is valid Go code
  * Generic code compiles and can be tested
  * Use `stdin` and `stdout` or specify in and out files
  * Supports Go 1.4's [go generate](http://tip.golang.org/doc/go1.4#gogenerate)
  * Multiple specific types will generate every permutation
  * Use `BUILTINS` and `NUMBERS` wildtype to generate specific code for all built-in (and number) Go types
  * Function names and comments also get updated

## Library

We have started building a [library of common things](https://github.com/cheekybits/gennylib), and you can use `genny get` to generate the specific versions you need.

For example: `genny get maps/concurrentmap.go "KeyType=BUILTINS ValueType=BUILTINS"` will print out generated code for all types for a concurrent map. Any file in the library may be generated locally in this way using all the same options given to `genny gen`.

## Usage

```
genny [{flags}] gen "{types}"

gen - generates type specific code from generic code.
get <package/file> - fetch a generic template from the online library and gen it.

{flags}  - (optional) Command line flags (see below)
{types}  - (required) Specific types for each generic type in the source
{types} format:  {generic}={specific}[,another][ {generic2}={specific2}]

Examples:
  Generic=Specific
  Generic1=Specific1 Generic2=Specific2
  Generic1=Specific1,Specific2 Generic2=Specific3,Specific4

Flags:
  -in="": file to parse instead of stdin
  -out="": file to save output to instead of stdout
  -pkg="": package name for generated files
```

  * Comma separated type lists will generate code for each type

### Flags

  * `-in` - specify the input file (rather than using stdin)
  * `-out` - specify the output file (rather than using stdout)

### go generate

To use Go 1.4's `go generate` capability, insert the following comment in your source code file:

```
//go:generate genny -in=$GOFILE -out=gen-$GOFILE gen "KeyType=string,int ValueType=string,int"
```

  * Start the line with `//go:generate `
  * Use the `-in` and `-out` flags to specify the files to work on
  * Use the `genny` command as usual after the flags

Now, running `go generate` (in a shell) for the package will cause the generic versions of the files to be generated.

  * The output file will be overwritten, so it's safe to call `go generate` many times
  * Use `$GOFILE` to refer to the current file
  * The `//go:generate` line will be removed from the output

To see a real example of how to use `genny` with `go generate`, look in the [example/go-generate directory](https://github.com/cheekybits/genny/tree/master/examples/go-generate).

## How it works

Define your generic types using the special `generic.Type` placeholder type:

```go
type KeyType generic.Type
type ValueType generic.Type
```

  * You can use as many as you like
  * Give them meaningful names

Then write the generic code referencing the types as your normally would:

```go
func SetValueTypeForKeyType(key KeyType, value ValueType) { /* ... */ }
```

  * Generic type names will also be replaced in comments and function names (see Real example below)

Since `generic.Type` is a real Go type, your code will compile, and you can even write unit tests against your generic code.

#### Generating specific versions

Pass the file through the `genny gen` tool with the specific types as the argument:

```
cat generic.go | genny gen "KeyType=string ValueType=interface{}"
```

The output will be the complete Go source file with the generic types replaced with the types specified in the arguments.

## Real example

Given [this generic Go code](https://github.com/cheekybits/genny/tree/master/examples/queue) which compiles and is tested:

```go
package queue

import "github.com/cheekybits/genny/generic"

// NOTE: this is how easy it is to define a generic type
type Something generic.Type

// SomethingQueue is a queue of Somethings.
type SomethingQueue struct {
  items []Something
}

func NewSomethingQueue() *SomethingQueue {
  return &SomethingQueue{items: make([]Something, 0)}
}
func (q *SomethingQueue) Push(item Something) {
  q.items = append(q.items, item)
}
func (q *SomethingQueue) Pop() Something {
  item := q.items[0]
  q.items = q.items[1:]
  return item
}
```

When `genny gen` is invoked like this:

```
cat source.go | genny gen "Something=string"
```

It outputs:

```go
// This file was automatically generated by genny.
// Any changes will be lost if this file is regenerated.
// see https://github.com/cheekybits/genny

package queue

// StringQueue is a queue of Strings.
type StringQueue struct {
  items []string
}

func NewStringQueue() *StringQueue {
  return &StringQueue{items: make([]string, 0)}
}
func (q *StringQueue) Push(item string) {
  q.items = append(q.items, item)
}
func (q *StringQueue) Pop() string {
  item := q.items[0]
  q.items = q.items[1:]
  return item
}
```

To get a _something_ for every built-in Go type plus one of your own types, you could run:

```
cat source.go | genny gen "Something=BUILTINS,*MyType"
```

#### More examples

Check out the [test code files](https://github.com/cheekybits/genny/tree/master/parse/test) for more real examples.

## Writing test code

Once you have defined a generic type with some code worth testing:

```go
package slice

import (
  "log"
  "reflect"

  "github.com/stretchr/gogen/generic"
)

type MyType generic.Type

func EnsureMyTypeSlice(objectOrSlice interface{}) []MyType {
  log.Printf("%v", reflect.TypeOf(objectOrSlice))
  switch obj := objectOrSlice.(type) {
  case []MyType:
    log.Println("  returning it untouched")
    return obj
  case MyType:
    log.Println("  wrapping in slice")
    return []MyType{obj}
  default:
    panic("ensure slice needs MyType or []MyType")
  }
}
```

You can treat it like any normal Go type in your test code:

```go
func TestEnsureMyTypeSlice(t *testing.T) {

  myType := new(MyType)
  slice := EnsureMyTypeSlice(myType)
  if assert.NotNil(t, slice) {
    assert.Equal(t, slice[0], myType)
  }

  slice = EnsureMyTypeSlice(slice)
  log.Printf("%#v", slice[0])
  if assert.NotNil(t, slice) {
    assert.Equal(t, slice[0], myType)
  }

}
```

### Understanding what `generic.Type` is

Because `generic.Type` is an empty interface type (literally `interface{}`) every other type will be considered to be a `generic.Type` if you are switching on the type of an object. Of course, once the specific versions are generated, this issue goes away but it's worth knowing when you are writing your tests against generic code.

### Contributions

  * See the [API documentation for the parse package](http://godoc.org/github.com/cheekybits/genny/parse)
  * Please do TDD
  * All input welcome
