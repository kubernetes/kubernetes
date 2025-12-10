randfill
======

randfill is a library for populating go objects with random values.

This is a fork of github.com/google/gofuzz, which was archived.

NOTE: This repo is supported only for use within Kubernetes.  It is not our
intention to support general use.  That said, if it works for you, that's
great!  If you have a problem, please feel free to file an issue, but be aware
that it may not be a priority for us to fix it unless it is affecting
Kubernetes.  PRs are welcome, within reason.

[![GoDoc](https://godoc.org/sigs.k8s.io/randfill?status.svg)](https://godoc.org/sigs.k8s.io/randfill)

This is useful for testing:

* Do your project's objects really serialize/unserialize correctly in all cases?
* Is there an incorrectly formatted object that will cause your project to panic?

Import with ```import "sigs.k8s.io/randfill"```

You can use it on single variables:
```go
f := randfill.New()
var myInt int
f.Fill(&myInt) // myInt gets a random value.
```

You can use it on maps:
```go
f := randfill.New().NilChance(0).NumElements(1, 1)
var myMap map[ComplexKeyType]string
f.Fill(&myMap) // myMap will have exactly one element.
```

Customize the chance of getting a nil pointer:
```go
f := randfill.New().NilChance(.5)
var fancyStruct struct {
  A, B, C, D *string
}
f.Fill(&fancyStruct) // About half the pointers should be set.
```

You can even customize the randomization completely if needed:
```go
type MyEnum string
const (
        A MyEnum = "A"
        B MyEnum = "B"
)
type MyInfo struct {
        Type MyEnum
        AInfo *string
        BInfo *string
}

f := randfill.New().NilChance(0).Funcs(
        func(e *MyInfo, c randfill.Continue) {
                switch c.Intn(2) {
                case 0:
                        e.Type = A
                        c.Fill(&e.AInfo)
                case 1:
                        e.Type = B
                        c.Fill(&e.BInfo)
                }
        },
)

var myObject MyInfo
f.Fill(&myObject) // Type will correspond to whether A or B info is set.
```

See more examples in ```example_test.go```.

## dvyukov/go-fuzz integration

You can use this library for easier [go-fuzz](https://github.com/dvyukov/go-fuzz)ing.
go-fuzz provides the user a byte-slice, which should be converted to different inputs
for the tested function. This library can help convert the byte slice. Consider for
example a fuzz test for a the function `mypackage.MyFunc` that takes an int arguments:
```go
// +build gofuzz
package mypackage

import "sigs.k8s.io/randfill"

func Fuzz(data []byte) int {
        var i int
        randfill.NewFromGoFuzz(data).Fill(&i)
        MyFunc(i)
        return 0
}
```

Happy testing!
