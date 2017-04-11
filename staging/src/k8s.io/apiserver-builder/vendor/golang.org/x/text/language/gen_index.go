// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// This file generates derivative tables based on the language package itself.

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")

	draft = flag.String("draft",
		"contributed",
		`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)
)

func main() {
	gen.Init()

	// Read the CLDR zip file.
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	w := gen.NewCodeWriter()
	defer func() {
		buf := &bytes.Buffer{}

		if _, err = w.WriteGo(buf, "language"); err != nil {
			log.Fatalf("Error formatting file index.go: %v", err)
		}

		// Since we're generating a table for our own package we need to rewrite
		// doing the equivalent of go fmt -r 'language.b -> b'. Using
		// bytes.Replace will do.
		out := bytes.Replace(buf.Bytes(), []byte("language."), nil, -1)
		if err := ioutil.WriteFile("index.go", out, 0600); err != nil {
			log.Fatalf("Could not create file index.go: %v", err)
		}
	}()

	m := map[language.Tag]bool{}
	for _, lang := range data.Locales() {
		// We include all locales unconditionally to be consistent with en_US.
		// We want en_US, even though it has no data associated with it.

		// TODO: put any of the languages for which no data exists at the end
		// of the index. This allows all components based on ICU to use that
		// as the cutoff point.
		// if x := data.RawLDML(lang); false ||
		// 	x.LocaleDisplayNames != nil ||
		// 	x.Characters != nil ||
		// 	x.Delimiters != nil ||
		// 	x.Measurement != nil ||
		// 	x.Dates != nil ||
		// 	x.Numbers != nil ||
		// 	x.Units != nil ||
		// 	x.ListPatterns != nil ||
		// 	x.Collations != nil ||
		// 	x.Segmentations != nil ||
		// 	x.Rbnf != nil ||
		// 	x.Annotations != nil ||
		// 	x.Metadata != nil {

		// TODO: support POSIX natively, albeit non-standard.
		tag := language.Make(strings.Replace(lang, "_POSIX", "-u-va-posix", 1))
		m[tag] = true
		// }
	}
	// Include locales for plural rules, which uses a different structure.
	for _, plurals := range data.Supplemental().Plurals {
		for _, rules := range plurals.PluralRules {
			for _, lang := range strings.Split(rules.Locales, " ") {
				m[language.Make(lang)] = true
			}
		}
	}

	var core, special []language.Tag

	for t := range m {
		if x := t.Extensions(); len(x) != 0 && fmt.Sprint(x) != "[u-va-posix]" {
			log.Fatalf("Unexpected extension %v in %v", x, t)
		}
		if len(t.Variants()) == 0 && len(t.Extensions()) == 0 {
			core = append(core, t)
		} else {
			special = append(special, t)
		}
	}

	w.WriteComment(`
	NumCompactTags is the number of common tags. The maximum tag is
	NumCompactTags-1.`)
	w.WriteConst("NumCompactTags", len(core)+len(special))

	sort.Sort(byAlpha(special))
	w.WriteVar("specialTags", special)

	// TODO: order by frequency?
	sort.Sort(byAlpha(core))

	// Size computations are just an estimate.
	w.Size += int(reflect.TypeOf(map[uint32]uint16{}).Size())
	w.Size += len(core) * 6 // size of uint32 and uint16

	fmt.Fprintln(w)
	fmt.Fprintln(w, "var coreTags = map[uint32]uint16{")
	fmt.Fprintln(w, "0x0: 0, // und")
	i := len(special) + 1 // Und and special tags already written.
	for _, t := range core {
		if t == language.Und {
			continue
		}
		fmt.Fprint(w.Hash, t, i)
		b, s, r := t.Raw()
		fmt.Fprintf(w, "0x%s%s%s: %d, // %s\n",
			getIndex(b, 3), // 3 is enough as it is guaranteed to be a compact number
			getIndex(s, 2),
			getIndex(r, 3),
			i, t)
		i++
	}
	fmt.Fprintln(w, "}")
}

// getIndex prints the subtag type and extracts its index of size nibble.
// If the index is less than n nibbles, the result is prefixed with 0s.
func getIndex(x interface{}, n int) string {
	s := fmt.Sprintf("%#v", x) // s is of form Type{typeID: 0x00}
	s = s[strings.Index(s, "0x")+2 : len(s)-1]
	return strings.Repeat("0", n-len(s)) + s
}

type byAlpha []language.Tag

func (a byAlpha) Len() int           { return len(a) }
func (a byAlpha) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byAlpha) Less(i, j int) bool { return a[i].String() < a[j].String() }
