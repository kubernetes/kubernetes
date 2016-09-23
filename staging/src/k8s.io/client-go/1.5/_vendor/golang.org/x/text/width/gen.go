// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program generates the trie for width operations. The generated table
// includes width category information as well as the normalization mappings.
package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"unicode/utf8"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/triegen"
)

// See gen_common.go for flags.

func main() {
	gen.Init()
	genTables()
	genTests()
	gen.Repackage("gen_trieval.go", "trieval.go", "width")
	gen.Repackage("gen_common.go", "common_test.go", "width")
}

func genTables() {
	t := triegen.NewTrie("width")
	// fold and inverse mappings. See mapComment for a description of the format
	// of each entry. Add dummy value to make an index of 0 mean no mapping.
	inverse := [][4]byte{{}}
	mapping := map[[4]byte]int{[4]byte{}: 0}

	getWidthData(func(r rune, tag elem, alt rune) {
		idx := 0
		if alt != 0 {
			var buf [4]byte
			buf[0] = byte(utf8.EncodeRune(buf[1:], alt))
			s := string(r)
			buf[buf[0]] ^= s[len(s)-1]
			var ok bool
			if idx, ok = mapping[buf]; !ok {
				idx = len(mapping)
				if idx > math.MaxUint8 {
					log.Fatalf("Index %d does not fit in a byte.", idx)
				}
				mapping[buf] = idx
				inverse = append(inverse, buf)
			}
		}
		t.Insert(r, uint64(tag|elem(idx)))
	})

	w := &bytes.Buffer{}
	gen.WriteUnicodeVersion(w)

	sz, err := t.Gen(w)
	if err != nil {
		log.Fatal(err)
	}

	sz += writeMappings(w, inverse)

	fmt.Fprintf(w, "// Total table size %d bytes (%dKiB)\n", sz, sz/1024)

	gen.WriteGoFile(*outputFile, "width", w.Bytes())
}

const inverseDataComment = `
// inverseData contains 4-byte entries of the following format:
//   <length> <modified UTF-8-encoded rune> <0 padding>
// The last byte of the UTF-8-encoded rune is xor-ed with the last byte of the
// UTF-8 encoding of the original rune. Mappings often have the following
// pattern:
//   Ａ -> A  (U+FF21 -> U+0041)
//   Ｂ -> B  (U+FF22 -> U+0042)
//   ...
// By xor-ing the last byte the same entry can be shared by many mappings. This
// reduces the total number of distinct entries by about two thirds.
// The resulting entry for the aforementioned mappings is
//   { 0x01, 0xE0, 0x00, 0x00 }
// Using this entry to map U+FF21 (UTF-8 [EF BC A1]), we get
//   E0 ^ A1 = 41.
// Similarly, for U+FF22 (UTF-8 [EF BC A2]), we get
//   E0 ^ A2 = 42.
// Note that because of the xor-ing, the byte sequence stored in the entry is
// not valid UTF-8.`

func writeMappings(w io.Writer, data [][4]byte) int {
	fmt.Fprintln(w, inverseDataComment)
	fmt.Fprintf(w, "var inverseData = [%d][4]byte{\n", len(data))
	for _, x := range data {
		fmt.Fprintf(w, "{ 0x%02x, 0x%02x, 0x%02x, 0x%02x },\n", x[0], x[1], x[2], x[3])
	}
	fmt.Fprintln(w, "}")
	return len(data) * 4
}

func genTests() {
	w := &bytes.Buffer{}
	fmt.Fprintf(w, "\nvar mapRunes = map[rune]struct{r rune; e elem}{\n")
	getWidthData(func(r rune, tag elem, alt rune) {
		if alt != 0 {
			fmt.Fprintf(w, "\t0x%X: {0x%X, 0x%X},\n", r, alt, tag)
		}
	})
	fmt.Fprintln(w, "}")
	gen.WriteGoFile("runes_test.go", "width", w.Bytes())
}
