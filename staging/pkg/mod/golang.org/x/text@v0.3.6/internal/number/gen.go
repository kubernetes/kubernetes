// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"flag"
	"fmt"
	"log"
	"reflect"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/language"
	"golang.org/x/text/internal/language/compact"
	"golang.org/x/text/internal/number"
	"golang.org/x/text/internal/stringset"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")
	outputFile     = flag.String("output", "tables.go", "output file")
	outputTestFile = flag.String("testoutput", "data_test.go", "output file")

	draft = flag.String("draft",
		"contributed",
		`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)
)

func main() {
	gen.Init()

	const pkg = "number"

	gen.Repackage("gen_common.go", "common.go", pkg)
	// Read the CLDR zip file.
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("numbers", "numberingSystem")
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	w := gen.NewCodeWriter()
	defer w.WriteGoFile(*outputFile, pkg)

	fmt.Fprintln(w, `import "golang.org/x/text/internal/stringset"`)

	gen.WriteCLDRVersion(w)

	genNumSystem(w, data)
	genSymbols(w, data)
	genFormats(w, data)
}

var systemMap = map[string]system{"latn": 0}

func getNumberSystem(str string) system {
	ns, ok := systemMap[str]
	if !ok {
		log.Fatalf("No index for numbering system %q", str)
	}
	return ns
}

func genNumSystem(w *gen.CodeWriter, data *cldr.CLDR) {
	numSysData := []systemData{
		{digitSize: 1, zero: [4]byte{'0'}},
	}

	for _, ns := range data.Supplemental().NumberingSystems.NumberingSystem {
		if len(ns.Digits) == 0 {
			continue
		}
		switch ns.Id {
		case "latn":
			// hard-wired
			continue
		case "hanidec":
			// non-consecutive digits: treat as "algorithmic"
			continue
		}

		zero, sz := utf8.DecodeRuneInString(ns.Digits)
		if ns.Digits[sz-1]+9 > 0xBF { // 1011 1111: highest continuation byte
			log.Fatalf("Last byte of zero value overflows for %s", ns.Id)
		}

		i := rune(0)
		for _, r := range ns.Digits {
			// Verify that we can do simple math on the UTF-8 byte sequence
			// of zero to get the digit.
			if zero+i != r {
				// Runes not consecutive.
				log.Fatalf("Digit %d of %s (%U) is not offset correctly from zero value", i, ns.Id, r)
			}
			i++
		}
		var x [utf8.UTFMax]byte
		utf8.EncodeRune(x[:], zero)
		id := system(len(numSysData))
		systemMap[ns.Id] = id
		numSysData = append(numSysData, systemData{
			id:        id,
			digitSize: byte(sz),
			zero:      x,
		})
	}
	w.WriteVar("numSysData", numSysData)

	algoID := system(len(numSysData))
	fmt.Fprintln(w, "const (")
	for _, ns := range data.Supplemental().NumberingSystems.NumberingSystem {
		id, ok := systemMap[ns.Id]
		if !ok {
			id = algoID
			systemMap[ns.Id] = id
			algoID++
		}
		fmt.Fprintf(w, "num%s = %#x\n", strings.Title(ns.Id), id)
	}
	fmt.Fprintln(w, "numNumberSystems")
	fmt.Fprintln(w, ")")

	fmt.Fprintln(w, "var systemMap = map[string]system{")
	for _, ns := range data.Supplemental().NumberingSystems.NumberingSystem {
		fmt.Fprintf(w, "%q: num%s,\n", ns.Id, strings.Title(ns.Id))
		w.Size += len(ns.Id) + 16 + 1 // very coarse approximation
	}
	fmt.Fprintln(w, "}")
}

func genSymbols(w *gen.CodeWriter, data *cldr.CLDR) {
	d, err := cldr.ParseDraft(*draft)
	if err != nil {
		log.Fatalf("invalid draft level: %v", err)
	}

	nNumberSystems := system(len(systemMap))

	type symbols [NumSymbolTypes]string

	type key struct {
		tag    compact.ID
		system system
	}
	symbolMap := map[key]*symbols{}

	defaults := map[compact.ID]system{}

	for _, lang := range data.Locales() {
		ldml := data.RawLDML(lang)
		if ldml.Numbers == nil {
			continue
		}
		langIndex, ok := compact.FromTag(language.MustParse(lang))
		if !ok {
			log.Fatalf("No compact index for language %s", lang)
		}
		if d := ldml.Numbers.DefaultNumberingSystem; len(d) > 0 {
			defaults[langIndex] = getNumberSystem(d[0].Data())
		}

		syms := cldr.MakeSlice(&ldml.Numbers.Symbols)
		syms.SelectDraft(d)

		getFirst := func(name string, x interface{}) string {
			v := reflect.ValueOf(x)
			slice := cldr.MakeSlice(x)
			slice.SelectAnyOf("alt", "", "alt")
			if reflect.Indirect(v).Len() == 0 {
				return ""
			} else if reflect.Indirect(v).Len() > 1 {
				log.Fatalf("%s: multiple values of %q within single symbol not supported.", lang, name)
			}
			return reflect.Indirect(v).Index(0).MethodByName("Data").Call(nil)[0].String()
		}

		for _, sym := range ldml.Numbers.Symbols {
			if sym.NumberSystem == "" {
				// This is just linking the default of root to "latn".
				continue
			}
			symbolMap[key{langIndex, getNumberSystem(sym.NumberSystem)}] = &symbols{
				SymDecimal:                getFirst("decimal", &sym.Decimal),
				SymGroup:                  getFirst("group", &sym.Group),
				SymList:                   getFirst("list", &sym.List),
				SymPercentSign:            getFirst("percentSign", &sym.PercentSign),
				SymPlusSign:               getFirst("plusSign", &sym.PlusSign),
				SymMinusSign:              getFirst("minusSign", &sym.MinusSign),
				SymExponential:            getFirst("exponential", &sym.Exponential),
				SymSuperscriptingExponent: getFirst("superscriptingExponent", &sym.SuperscriptingExponent),
				SymPerMille:               getFirst("perMille", &sym.PerMille),
				SymInfinity:               getFirst("infinity", &sym.Infinity),
				SymNan:                    getFirst("nan", &sym.Nan),
				SymTimeSeparator:          getFirst("timeSeparator", &sym.TimeSeparator),
			}
		}
	}

	// Expand all values.
	for k, syms := range symbolMap {
		for t := SymDecimal; t < NumSymbolTypes; t++ {
			p := k.tag
			for syms[t] == "" {
				p = p.Parent()
				if pSyms, ok := symbolMap[key{p, k.system}]; ok && (*pSyms)[t] != "" {
					syms[t] = (*pSyms)[t]
					break
				}
				if p == 0 /* und */ {
					// Default to root, latn.
					syms[t] = (*symbolMap[key{}])[t]
				}
			}
		}
	}

	// Unique the symbol sets and write the string data.
	m := map[symbols]int{}
	sb := stringset.NewBuilder()

	symIndex := [][NumSymbolTypes]byte{}

	for ns := system(0); ns < nNumberSystems; ns++ {
		for _, l := range data.Locales() {
			langIndex, _ := compact.FromTag(language.MustParse(l))
			s := symbolMap[key{langIndex, ns}]
			if s == nil {
				continue
			}
			if _, ok := m[*s]; !ok {
				m[*s] = len(symIndex)
				sb.Add(s[:]...)
				var x [NumSymbolTypes]byte
				for i := SymDecimal; i < NumSymbolTypes; i++ {
					x[i] = byte(sb.Index((*s)[i]))
				}
				symIndex = append(symIndex, x)
			}
		}
	}
	w.WriteVar("symIndex", symIndex)
	w.WriteVar("symData", sb.Set())

	// resolveSymbolIndex gets the index from the closest matching locale,
	// including the locale itself.
	resolveSymbolIndex := func(langIndex compact.ID, ns system) symOffset {
		for {
			if sym := symbolMap[key{langIndex, ns}]; sym != nil {
				return symOffset(m[*sym])
			}
			if langIndex == 0 {
				return 0 // und, latn
			}
			langIndex = langIndex.Parent()
		}
	}

	// Create an index with the symbols for each locale for the latn numbering
	// system. If this is not the default, or the only one, for a locale, we
	// will overwrite the value later.
	var langToDefaults [compact.NumCompactTags]symOffset
	for _, l := range data.Locales() {
		langIndex, _ := compact.FromTag(language.MustParse(l))
		langToDefaults[langIndex] = resolveSymbolIndex(langIndex, 0)
	}

	// Delete redundant entries.
	for _, l := range data.Locales() {
		langIndex, _ := compact.FromTag(language.MustParse(l))
		def := defaults[langIndex]
		syms := symbolMap[key{langIndex, def}]
		if syms == nil {
			continue
		}
		for ns := system(0); ns < nNumberSystems; ns++ {
			if ns == def {
				continue
			}
			if altSyms, ok := symbolMap[key{langIndex, ns}]; ok && *altSyms == *syms {
				delete(symbolMap, key{langIndex, ns})
			}
		}
	}

	// Create a sorted list of alternatives per language. This will only need to
	// be referenced if a user specified an alternative numbering system.
	var langToAlt []altSymData
	for _, l := range data.Locales() {
		langIndex, _ := compact.FromTag(language.MustParse(l))
		start := len(langToAlt)
		if start >= hasNonLatnMask {
			log.Fatalf("Number of alternative assignments >= %x", hasNonLatnMask)
		}
		// Create the entry for the default value.
		def := defaults[langIndex]
		langToAlt = append(langToAlt, altSymData{
			compactTag: langIndex,
			system:     def,
			symIndex:   resolveSymbolIndex(langIndex, def),
		})

		for ns := system(0); ns < nNumberSystems; ns++ {
			if def == ns {
				continue
			}
			if sym := symbolMap[key{langIndex, ns}]; sym != nil {
				langToAlt = append(langToAlt, altSymData{
					compactTag: langIndex,
					system:     ns,
					symIndex:   resolveSymbolIndex(langIndex, ns),
				})
			}
		}
		if def == 0 && len(langToAlt) == start+1 {
			// No additional data: erase the entry.
			langToAlt = langToAlt[:start]
		} else {
			// Overwrite the entry in langToDefaults.
			langToDefaults[langIndex] = hasNonLatnMask | symOffset(start)
		}
	}
	w.WriteComment(`
langToDefaults maps a compact language index to the default numbering system
and default symbol set`)
	w.WriteVar("langToDefaults", langToDefaults)

	w.WriteComment(`
langToAlt is a list of numbering system and symbol set pairs, sorted and
marked by compact language index.`)
	w.WriteVar("langToAlt", langToAlt)
}

// genFormats generates the lookup table for decimal, scientific and percent
// patterns.
//
// CLDR allows for patterns to be different per language for different numbering
// systems. In practice the patterns are set to be consistent for a language
// independent of the numbering system. genFormats verifies that no language
// deviates from this.
func genFormats(w *gen.CodeWriter, data *cldr.CLDR) {
	d, err := cldr.ParseDraft(*draft)
	if err != nil {
		log.Fatalf("invalid draft level: %v", err)
	}

	// Fill the first slot with a dummy so we can identify unspecified tags.
	formats := []number.Pattern{{}}
	patterns := map[string]int{}

	// TODO: It would be possible to eliminate two of these slices by having
	// another indirection and store a reference to the combination of patterns.
	decimal := make([]byte, compact.NumCompactTags)
	scientific := make([]byte, compact.NumCompactTags)
	percent := make([]byte, compact.NumCompactTags)

	for _, lang := range data.Locales() {
		ldml := data.RawLDML(lang)
		if ldml.Numbers == nil {
			continue
		}
		langIndex, ok := compact.FromTag(language.MustParse(lang))
		if !ok {
			log.Fatalf("No compact index for language %s", lang)
		}
		type patternSlice []*struct {
			cldr.Common
			Numbers string `xml:"numbers,attr"`
			Count   string `xml:"count,attr"`
		}

		add := func(name string, tags []byte, ps patternSlice) {
			sl := cldr.MakeSlice(&ps)
			sl.SelectDraft(d)
			if len(ps) == 0 {
				return
			}
			if len(ps) > 2 || len(ps) == 2 && ps[0] != ps[1] {
				log.Fatalf("Inconsistent %d patterns for language %s", name, lang)
			}
			s := ps[0].Data()

			index, ok := patterns[s]
			if !ok {
				nf, err := number.ParsePattern(s)
				if err != nil {
					log.Fatal(err)
				}
				index = len(formats)
				patterns[s] = index
				formats = append(formats, *nf)
			}
			tags[langIndex] = byte(index)
		}

		for _, df := range ldml.Numbers.DecimalFormats {
			for _, l := range df.DecimalFormatLength {
				if l.Type != "" {
					continue
				}
				for _, f := range l.DecimalFormat {
					add("decimal", decimal, f.Pattern)
				}
			}
		}
		for _, df := range ldml.Numbers.ScientificFormats {
			for _, l := range df.ScientificFormatLength {
				if l.Type != "" {
					continue
				}
				for _, f := range l.ScientificFormat {
					add("scientific", scientific, f.Pattern)
				}
			}
		}
		for _, df := range ldml.Numbers.PercentFormats {
			for _, l := range df.PercentFormatLength {
				if l.Type != "" {
					continue
				}
				for _, f := range l.PercentFormat {
					add("percent", percent, f.Pattern)
				}
			}
		}
	}

	// Complete the parent tag array to reflect inheritance. An index of 0
	// indicates an unspecified value.
	for _, data := range [][]byte{decimal, scientific, percent} {
		for i := range data {
			p := compact.ID(i)
			for ; data[p] == 0; p = p.Parent() {
			}
			data[i] = data[p]
		}
	}
	w.WriteVar("tagToDecimal", decimal)
	w.WriteVar("tagToScientific", scientific)
	w.WriteVar("tagToPercent", percent)

	value := strings.Replace(fmt.Sprintf("%#v", formats), "number.", "", -1)
	// Break up the lines. This won't give ideal perfect formatting, but it is
	// better than one huge line.
	value = strings.Replace(value, ", ", ",\n", -1)
	fmt.Fprintf(w, "var formats = %s\n", value)
}
