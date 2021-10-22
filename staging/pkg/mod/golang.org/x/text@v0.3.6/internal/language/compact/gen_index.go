// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

// This file generates derivative tables based on the language package itself.

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"golang.org/x/text/internal/language"
)

// Compact indices:
// Note -va-X variants only apply to localization variants.
// BCP variants only ever apply to language.
// The only ambiguity between tags is with regions.

func (b *builder) writeCompactIndex() {
	// Collect all language tags for which we have any data in CLDR.
	m := map[language.Tag]bool{}
	for _, lang := range b.data.Locales() {
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

	// TODO: plural rules are also defined for the deprecated tags:
	//    iw mo sh tl
	// Consider removing these as compact tags.

	// Include locales for plural rules, which uses a different structure.
	for _, plurals := range b.supp.Plurals {
		for _, rules := range plurals.PluralRules {
			for _, lang := range strings.Split(rules.Locales, " ") {
				m[language.Make(lang)] = true
			}
		}
	}

	var coreTags []language.CompactCoreInfo
	var special []string

	for t := range m {
		if x := t.Extensions(); len(x) != 0 && fmt.Sprint(x) != "[u-va-posix]" {
			log.Fatalf("Unexpected extension %v in %v", x, t)
		}
		if len(t.Variants()) == 0 && len(t.Extensions()) == 0 {
			cci, ok := language.GetCompactCore(t)
			if !ok {
				log.Fatalf("Locale for non-basic language %q", t)
			}
			coreTags = append(coreTags, cci)
		} else {
			special = append(special, t.String())
		}
	}

	w := b.w

	sort.Slice(coreTags, func(i, j int) bool { return coreTags[i] < coreTags[j] })
	sort.Strings(special)

	w.WriteComment(`
	NumCompactTags is the number of common tags. The maximum tag is
	NumCompactTags-1.`)
	w.WriteConst("NumCompactTags", len(m))

	fmt.Fprintln(w, "const (")
	for i, t := range coreTags {
		fmt.Fprintf(w, "%s ID = %d\n", ident(t.Tag().String()), i)
	}
	for i, t := range special {
		fmt.Fprintf(w, "%s ID = %d\n", ident(t), i+len(coreTags))
	}
	fmt.Fprintln(w, ")")

	w.WriteVar("coreTags", coreTags)

	w.WriteConst("specialTagsStr", strings.Join(special, " "))
}

func ident(s string) string {
	return strings.Replace(s, "-", "", -1) + "Index"
}
