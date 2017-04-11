// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"log"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

func main() {
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	w := gen.NewCodeWriter()
	defer w.WriteGoFile("tables.go", "internal")

	// Create parents table.
	parents := make([]uint16, language.NumCompactTags)
	for _, loc := range data.Locales() {
		tag := language.MustParse(loc)
		index, ok := language.CompactIndex(tag)
		if !ok {
			continue
		}
		parentIndex := 0 // und
		for p := tag.Parent(); p != language.Und; p = p.Parent() {
			if x, ok := language.CompactIndex(p); ok {
				parentIndex = x
				break
			}
		}
		parents[index] = uint16(parentIndex)
	}

	w.WriteComment(`
	Parent maps a compact index of a tag to the compact index of the parent of
	this tag.`)
	w.WriteVar("Parent", parents)
}
