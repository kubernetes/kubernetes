// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package triegen_test

import (
	"fmt"
	"io"
	"io/ioutil"

	"golang.org/x/text/internal/triegen"
)

func ExampleCompacter() {
	t := triegen.NewTrie("root")
	for r := rune(0); r < 10000; r += 64 {
		t.Insert(r, 0x9015BADA55^uint64(r))
	}
	sz, _ := t.Gen(ioutil.Discard)

	fmt.Printf("Size normal:    %5d\n", sz)

	var c myCompacter
	sz, _ = t.Gen(ioutil.Discard, triegen.Compact(&c))

	fmt.Printf("Size compacted: %5d\n", sz)

	// Output:
	// Size normal:    81344
	// Size compacted:  3224
}

// A myCompacter accepts a block if only the first value is given.
type myCompacter []uint64

func (c *myCompacter) Size(values []uint64) (sz int, ok bool) {
	for _, v := range values[1:] {
		if v != 0 {
			return 0, false
		}
	}
	return 8, true // the size of a uint64
}

func (c *myCompacter) Store(v []uint64) uint32 {
	x := uint32(len(*c))
	*c = append(*c, v[0])
	return x
}

func (c *myCompacter) Print(w io.Writer) error {
	fmt.Fprintln(w, "var firstValue = []uint64{")
	for _, v := range *c {
		fmt.Fprintf(w, "\t%#x,\n", v)
	}
	fmt.Fprintln(w, "}")
	return nil
}

func (c *myCompacter) Handler() string {
	return "getFirstValue"

	// Where getFirstValue is included along with the generated code:
	// func getFirstValue(n uint32, b byte) uint64 {
	//     if b == 0x80 { // the first continuation byte
	//         return firstValue[n]
	//     }
	//     return 0
	//  }
}
