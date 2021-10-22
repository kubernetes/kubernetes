// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// Language tag table generator.
// Data read from the web.

package main

import (
	"flag"
	"fmt"
	"log"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test",
		false,
		"test existing tables; can be used to compare web data with package data.")
	outputFile = flag.String("output",
		"tables.go",
		"output file for generated tables")
)

func main() {
	gen.Init()

	w := gen.NewCodeWriter()
	defer w.WriteGoFile("tables.go", "compact")

	fmt.Fprintln(w, `import "golang.org/x/text/internal/language"`)

	b := newBuilder(w)
	gen.WriteCLDRVersion(w)

	b.writeCompactIndex()
}

type builder struct {
	w    *gen.CodeWriter
	data *cldr.CLDR
	supp *cldr.SupplementalData
}

func newBuilder(w *gen.CodeWriter) *builder {
	r := gen.OpenCLDRCoreZip()
	defer r.Close()
	d := &cldr.Decoder{}
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatal(err)
	}
	b := builder{
		w:    w,
		data: data,
		supp: data.Supplemental(),
	}
	return &b
}
