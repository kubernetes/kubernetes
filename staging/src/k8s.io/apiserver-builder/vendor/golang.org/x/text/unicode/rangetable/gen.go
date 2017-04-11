// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"reflect"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/ucd"
	"golang.org/x/text/unicode/rangetable"
)

var versionList = flag.String("versions", "",
	"list of versions for which to generate RangeTables")

const bootstrapMessage = `No versions specified.
To bootstrap the code generation, run:
	go run gen.go --versions=4.1.0,5.0.0,6.0.0,6.1.0,6.2.0,6.3.0,7.0.0

and ensure that the latest versions are included by checking:
	http://www.unicode.org/Public/`

func getVersions() []string {
	if *versionList == "" {
		log.Fatal(bootstrapMessage)
	}

	versions := strings.Split(*versionList, ",")
	sort.Strings(versions)

	// Ensure that at least the current version is included.
	for _, v := range versions {
		if v == gen.UnicodeVersion() {
			return versions
		}
	}

	versions = append(versions, gen.UnicodeVersion())
	sort.Strings(versions)
	return versions
}

func main() {
	gen.Init()

	versions := getVersions()

	w := &bytes.Buffer{}

	fmt.Fprintf(w, "//go:generate go run gen.go --versions=%s\n\n", strings.Join(versions, ","))
	fmt.Fprintf(w, "import \"unicode\"\n\n")

	vstr := func(s string) string { return strings.Replace(s, ".", "_", -1) }

	fmt.Fprintf(w, "var assigned = map[string]*unicode.RangeTable{\n")
	for _, v := range versions {
		fmt.Fprintf(w, "\t%q: assigned%s,\n", v, vstr(v))
	}
	fmt.Fprintf(w, "}\n\n")

	var size int
	for _, v := range versions {
		assigned := []rune{}

		r := gen.Open("http://www.unicode.org/Public/", "", v+"/ucd/UnicodeData.txt")
		ucd.Parse(r, func(p *ucd.Parser) {
			assigned = append(assigned, p.Rune(0))
		})

		rt := rangetable.New(assigned...)
		sz := int(reflect.TypeOf(unicode.RangeTable{}).Size())
		sz += int(reflect.TypeOf(unicode.Range16{}).Size()) * len(rt.R16)
		sz += int(reflect.TypeOf(unicode.Range32{}).Size()) * len(rt.R32)

		fmt.Fprintf(w, "// size %d bytes (%d KiB)\n", sz, sz/1024)
		fmt.Fprintf(w, "var assigned%s = ", vstr(v))
		print(w, rt)

		size += sz
	}

	fmt.Fprintf(w, "// Total size %d bytes (%d KiB)\n", size, size/1024)

	gen.WriteGoFile("tables.go", "rangetable", w.Bytes())
}

func print(w io.Writer, rt *unicode.RangeTable) {
	fmt.Fprintln(w, "&unicode.RangeTable{")
	fmt.Fprintln(w, "\tR16: []unicode.Range16{")
	for _, r := range rt.R16 {
		fmt.Fprintf(w, "\t\t{%#04x, %#04x, %d},\n", r.Lo, r.Hi, r.Stride)
	}
	fmt.Fprintln(w, "\t},")
	fmt.Fprintln(w, "\tR32: []unicode.Range32{")
	for _, r := range rt.R32 {
		fmt.Fprintf(w, "\t\t{%#08x, %#08x, %d},\n", r.Lo, r.Hi, r.Stride)
	}
	fmt.Fprintln(w, "\t},")
	fmt.Fprintf(w, "\tLatinOffset: %d,\n", rt.LatinOffset)
	fmt.Fprintf(w, "}\n\n")
}
