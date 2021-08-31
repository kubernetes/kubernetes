# bitset

*Go language library to map between non-negative integers and boolean values*

[![Test](https://github.com/bits-and-blooms/bitset/workflows/Test/badge.svg)](https://github.com/willf/bitset/actions?query=workflow%3ATest)
[![Go Report Card](https://goreportcard.com/badge/github.com/willf/bitset)](https://goreportcard.com/report/github.com/willf/bitset)
[![PkgGoDev](https://pkg.go.dev/badge/github.com/bits-and-blooms/bitset?tab=doc)](https://pkg.go.dev/github.com/bits-and-blooms/bitset?tab=doc)


## Description

Package bitset implements bitsets, a mapping between non-negative integers and boolean values.
It should be more efficient than map[uint] bool.

It provides methods for setting, clearing, flipping, and testing individual integers.

But it also provides set intersection, union, difference, complement, and symmetric operations, as well as tests to check whether any, all, or no bits are set, and querying a bitset's current length and number of positive bits.

BitSets are expanded to the size of the largest set bit; the memory allocation is approximately Max bits, where Max is the largest set bit. BitSets are never shrunk. On creation, a hint can be given for the number of bits that will be used.

Many of the methods, including Set, Clear, and Flip, return a BitSet pointer, which allows for chaining.

### Example use:

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/bits-and-blooms/bitset"
)

func main() {
	fmt.Printf("Hello from BitSet!\n")
	var b bitset.BitSet
	// play some Go Fish
	for i := 0; i < 100; i++ {
		card1 := uint(rand.Intn(52))
		card2 := uint(rand.Intn(52))
		b.Set(card1)
		if b.Test(card2) {
			fmt.Println("Go Fish!")
		}
		b.Clear(card1)
	}

	// Chaining
	b.Set(10).Set(11)

	for i, e := b.NextSet(0); e; i, e = b.NextSet(i + 1) {
		fmt.Println("The following bit is set:", i)
	}
	if b.Intersection(bitset.New(100).Set(10)).Count() == 1 {
		fmt.Println("Intersection works.")
	} else {
		fmt.Println("Intersection doesn't work???")
	}
}
```

As an alternative to BitSets, one should check out the 'big' package, which provides a (less set-theoretical) view of bitsets.

Package documentation is at: https://pkg.go.dev/github.com/bits-and-blooms/bitset?tab=doc

## Memory Usage

The memory usage of a bitset using N bits is at least N/8 bytes. The number of bits in a bitset is at least as large as one plus the greatest bit index you have accessed. Thus it is possible to run out of memory while using a bitset. If you have lots of bits, you might prefer compressed bitsets, like the [Roaring bitmaps](http://roaringbitmap.org) and its [Go implementation](https://github.com/RoaringBitmap/roaring).

## Implementation Note

Go 1.9 introduced a native `math/bits` library. We provide backward compatibility to Go 1.7, which might be removed.

It is possible that a later version will match the `math/bits` return signature for counts (which is `int`, rather than our library's `unit64`). If so, the version will be bumped.

## Installation

```bash
go get github.com/bits-and-blooms/bitset
```

## Contributing

If you wish to contribute to this project, please branch and issue a pull request against master ("[GitHub Flow](https://guides.github.com/introduction/flow/)")

## Running all tests

Before committing the code, please check if it passes tests, has adequate coverage, etc.
```bash
go test
go test -cover
```
