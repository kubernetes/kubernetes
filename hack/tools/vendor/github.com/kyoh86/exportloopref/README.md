# exportloopref

An analyzer that finds exporting pointers for loop variables.

[![PkgGoDev](https://pkg.go.dev/badge/kyoh86/exportloopref)](https://pkg.go.dev/kyoh86/exportloopref)
[![Go Report Card](https://goreportcard.com/badge/github.com/kyoh86/exportloopref)](https://goreportcard.com/report/github.com/kyoh86/exportloopref)
[![Coverage Status](https://img.shields.io/codecov/c/github/kyoh86/exportloopref.svg)](https://codecov.io/gh/kyoh86/exportloopref)
[![Release](https://github.com/kyoh86/exportloopref/workflows/Release/badge.svg)](https://github.com/kyoh86/exportloopref/releases)

## What's this?

Sample problem code from: https://github.com/kyoh86/exportloopref/blob/master/testdata/src/simple/simple.go

```go
package main

func main() {
	var intArray [4]*int
	var intSlice []*int
	var intRef *int
	var intStr struct{ x *int }

	println("loop expecting 10, 11, 12, 13")
	for i, p := range []int{10, 11, 12, 13} {
		printp(&p)                      // not a diagnostic
		intSlice = append(intSlice, &p) // want "exporting a pointer for the loop variable p"
		intArray[i] = &p                // want "exporting a pointer for the loop variable p"
		if i%2 == 0 {
			intRef = &p   // want "exporting a pointer for the loop variable p"
			intStr.x = &p // want "exporting a pointer for the loop variable p"
		}
		var vStr struct{ x *int }
		var vArray [4]*int
		var v *int
		if i%2 == 0 {
			v = &p         // not a diagnostic (x is local variable)
			vArray[1] = &p // not a diagnostic (x is local variable)
			vStr.x = &p
		}
		_ = v
	}

	println(`slice expecting "10, 11, 12, 13" but "13, 13, 13, 13"`)
	for _, p := range intSlice {
		printp(p)
	}
	println(`array expecting "10, 11, 12, 13" but "13, 13, 13, 13"`)
	for _, p := range intArray {
		printp(p)
	}
	println(`captured value expecting "12" but "13"`)
	printp(intRef)
}

func printp(p *int) {
	println(*p)
}
```

In Go, the `p` variable in the above loops is actually a single variable.
So in many case (like the above), using it makes for us annoying bugs.

You can find them with `exportloopref`, and fix it.

```go
package main

func main() {
	var intArray [4]*int
	var intSlice []*int
	var intRef *int
	var intStr struct{ x *int }

	println("loop expecting 10, 11, 12, 13")
	for i, p := range []int{10, 11, 12, 13} {
    p := p                          // FIX variable into the local variable
		printp(&p)
		intSlice = append(intSlice, &p) 
		intArray[i] = &p
		if i%2 == 0 {
			intRef = &p
			intStr.x = &p
		}
		var vStr struct{ x *int }
		var vArray [4]*int
		var v *int
		if i%2 == 0 {
			v = &p
			vArray[1] = &p
			vStr.x = &p
		}
		_ = v
	}

	println(`slice expecting "10, 11, 12, 13"`)
	for _, p := range intSlice {
		printp(p)
	}
	println(`array expecting "10, 11, 12, 13"`)
	for _, p := range intArray {
		printp(p)
	}
	println(`captured value expecting "12"`)
	printp(intRef)
}

func printp(p *int) {
	println(*p)
}
```

ref: https://github.com/kyoh86/exportloopref/blob/master/testdata/src/fixed/fixed.go

## Sensing policy

I want to make exportloopref as accurately as possible.
So some cases of lints will be false-negative.

e.g.

```go
var s Foo
for _, p := []int{10, 11, 12, 13} {
  s.Bar(&p) // If s stores the pointer, it will be bug.
}
```

If you want to report all of lints (with some false-positives),
you should use [looppointer](https://github.com/kyoh86/looppointer).

### Known false negatives

Case 1: pass the pointer to function to export.

Case 2: pass the pointer to local variable, and export it.

```go
package main

type List []*int

func (l *List) AppendP(p *int) {
	*l = append(*l, p)
}

func main() {
	var slice []*int
	list := List{}

	println("loop expect exporting 10, 11, 12, 13")
	for _, v := range []int{10, 11, 12, 13} {
		list.AppendP(&v) // Case 1: wanted "exporting a pointer for the loop variable v", but cannot be found

		p := &v                  // p is the local variable
		slice = append(slice, p) // Case 2: wanted "exporting a pointer for the loop variable v", but cannot be found
	}

	println(`slice expecting "10, 11, 12, 13" but "13, 13, 13, 13"`)
	for _, p := range slice {
		printp(p)
	}
	println(`array expecting "10, 11, 12, 13" but "13, 13, 13, 13"`)
	for _, p := range ([]*int)(list) {
		printp(p)
	}
}

func printp(p *int) {
	println(*p)
}
```

## Install

go:

```console
$ go get github.com/kyoh86/exportloopref/cmd/exportloopref
```

[homebrew](https://brew.sh/):

```console
$ brew install kyoh86/tap/exportloopref
```

[gordon](https://github.com/kyoh86/gordon):

```console
$ gordon install kyoh86/exportloopref
```

## Usage

```
exportloopref [-flag] [package]
```

### Flags

| Flag | Description |
| --- | --- |
| -V                 | print version and exit |
| -all               | no effect (deprecated) |
| -c int             | display offending line with this many lines of context (default -1) |
| -cpuprofile string | write CPU profile to this file |
| -debug string      | debug flags, any subset of "fpstv" |
| -fix               | apply all suggested fixes |
| -flags             | print analyzer flags in JSON |
| -json              | emit JSON output |
| -memprofile string | write memory profile to this file |
| -source            | no effect (deprecated) |
| -tags string       | no effect (deprecated) |
| -trace string      | write trace log to this file |
| -v                 | no effect (deprecated) |

# LICENSE

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg)](http://www.opensource.org/licenses/MIT)

This is distributed under the [MIT License](http://www.opensource.org/licenses/MIT).
