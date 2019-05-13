// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Language tag table generator.
// Data read from the web.

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/language"
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
	defer w.WriteGoFile("tables.go", "language")

	b := newBuilder(w)
	gen.WriteCLDRVersion(w)

	b.writeConstants()
	b.writeMatchData()
}

type builder struct {
	w    *gen.CodeWriter
	hw   io.Writer // MultiWriter for w and w.Hash
	data *cldr.CLDR
	supp *cldr.SupplementalData
}

func (b *builder) langIndex(s string) uint16 {
	return uint16(language.MustParseBase(s))
}

func (b *builder) regionIndex(s string) int {
	return int(language.MustParseRegion(s))
}

func (b *builder) scriptIndex(s string) int {
	return int(language.MustParseScript(s))
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
		hw:   io.MultiWriter(w, w.Hash),
		data: data,
		supp: data.Supplemental(),
	}
	return &b
}

// writeConsts computes f(v) for all v in values and writes the results
// as constants named _v to a single constant block.
func (b *builder) writeConsts(f func(string) int, values ...string) {
	fmt.Fprintln(b.w, "const (")
	for _, v := range values {
		fmt.Fprintf(b.w, "\t_%s = %v\n", v, f(v))
	}
	fmt.Fprintln(b.w, ")")
}

// TODO: region inclusion data will probably not be use used in future matchers.

var langConsts = []string{
	"de", "en", "fr", "it", "mo", "no", "nb", "pt", "sh", "mul", "und",
}

var scriptConsts = []string{
	"Latn", "Hani", "Hans", "Hant", "Qaaa", "Qaai", "Qabx", "Zinh", "Zyyy",
	"Zzzz",
}

var regionConsts = []string{
	"001", "419", "BR", "CA", "ES", "GB", "MD", "PT", "UK", "US",
	"ZZ", "XA", "XC", "XK", // Unofficial tag for Kosovo.
}

func (b *builder) writeConstants() {
	b.writeConsts(func(s string) int { return int(b.langIndex(s)) }, langConsts...)
	b.writeConsts(b.regionIndex, regionConsts...)
	b.writeConsts(b.scriptIndex, scriptConsts...)
}

type mutualIntelligibility struct {
	want, have uint16
	distance   uint8
	oneway     bool
}

type scriptIntelligibility struct {
	wantLang, haveLang     uint16
	wantScript, haveScript uint8
	distance               uint8
	// Always oneway
}

type regionIntelligibility struct {
	lang     uint16 // compact language id
	script   uint8  // 0 means any
	group    uint8  // 0 means any; if bit 7 is set it means inverse
	distance uint8
	// Always twoway.
}

// writeMatchData writes tables with languages and scripts for which there is
// mutual intelligibility. The data is based on CLDR's languageMatching data.
// Note that we use a different algorithm than the one defined by CLDR and that
// we slightly modify the data. For example, we convert scores to confidence levels.
// We also drop all region-related data as we use a different algorithm to
// determine region equivalence.
func (b *builder) writeMatchData() {
	lm := b.supp.LanguageMatching.LanguageMatches
	cldr.MakeSlice(&lm).SelectAnyOf("type", "written_new")

	regionHierarchy := map[string][]string{}
	for _, g := range b.supp.TerritoryContainment.Group {
		regions := strings.Split(g.Contains, " ")
		regionHierarchy[g.Type] = append(regionHierarchy[g.Type], regions...)
	}
	regionToGroups := make([]uint8, language.NumRegions)

	idToIndex := map[string]uint8{}
	for i, mv := range lm[0].MatchVariable {
		if i > 6 {
			log.Fatalf("Too many groups: %d", i)
		}
		idToIndex[mv.Id] = uint8(i + 1)
		// TODO: also handle '-'
		for _, r := range strings.Split(mv.Value, "+") {
			todo := []string{r}
			for k := 0; k < len(todo); k++ {
				r := todo[k]
				regionToGroups[b.regionIndex(r)] |= 1 << uint8(i)
				todo = append(todo, regionHierarchy[r]...)
			}
		}
	}
	b.w.WriteVar("regionToGroups", regionToGroups)

	// maps language id to in- and out-of-group region.
	paradigmLocales := [][3]uint16{}
	locales := strings.Split(lm[0].ParadigmLocales[0].Locales, " ")
	for i := 0; i < len(locales); i += 2 {
		x := [3]uint16{}
		for j := 0; j < 2; j++ {
			pc := strings.SplitN(locales[i+j], "-", 2)
			x[0] = b.langIndex(pc[0])
			if len(pc) == 2 {
				x[1+j] = uint16(b.regionIndex(pc[1]))
			}
		}
		paradigmLocales = append(paradigmLocales, x)
	}
	b.w.WriteVar("paradigmLocales", paradigmLocales)

	b.w.WriteType(mutualIntelligibility{})
	b.w.WriteType(scriptIntelligibility{})
	b.w.WriteType(regionIntelligibility{})

	matchLang := []mutualIntelligibility{}
	matchScript := []scriptIntelligibility{}
	matchRegion := []regionIntelligibility{}
	// Convert the languageMatch entries in lists keyed by desired language.
	for _, m := range lm[0].LanguageMatch {
		// Different versions of CLDR use different separators.
		desired := strings.Replace(m.Desired, "-", "_", -1)
		supported := strings.Replace(m.Supported, "-", "_", -1)
		d := strings.Split(desired, "_")
		s := strings.Split(supported, "_")
		if len(d) != len(s) {
			log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
			continue
		}
		distance, _ := strconv.ParseInt(m.Distance, 10, 8)
		switch len(d) {
		case 2:
			if desired == supported && desired == "*_*" {
				continue
			}
			// language-script pair.
			matchScript = append(matchScript, scriptIntelligibility{
				wantLang:   uint16(b.langIndex(d[0])),
				haveLang:   uint16(b.langIndex(s[0])),
				wantScript: uint8(b.scriptIndex(d[1])),
				haveScript: uint8(b.scriptIndex(s[1])),
				distance:   uint8(distance),
			})
			if m.Oneway != "true" {
				matchScript = append(matchScript, scriptIntelligibility{
					wantLang:   uint16(b.langIndex(s[0])),
					haveLang:   uint16(b.langIndex(d[0])),
					wantScript: uint8(b.scriptIndex(s[1])),
					haveScript: uint8(b.scriptIndex(d[1])),
					distance:   uint8(distance),
				})
			}
		case 1:
			if desired == supported && desired == "*" {
				continue
			}
			if distance == 1 {
				// nb == no is already handled by macro mapping. Check there
				// really is only this case.
				if d[0] != "no" || s[0] != "nb" {
					log.Fatalf("unhandled equivalence %s == %s", s[0], d[0])
				}
				continue
			}
			// TODO: consider dropping oneway field and just doubling the entry.
			matchLang = append(matchLang, mutualIntelligibility{
				want:     uint16(b.langIndex(d[0])),
				have:     uint16(b.langIndex(s[0])),
				distance: uint8(distance),
				oneway:   m.Oneway == "true",
			})
		case 3:
			if desired == supported && desired == "*_*_*" {
				continue
			}
			if desired != supported {
				// This is now supported by CLDR, but only one case, which
				// should already be covered by paradigm locales. For instance,
				// test case "und, en, en-GU, en-IN, en-GB ; en-ZA ; en-GB" in
				// testdata/CLDRLocaleMatcherTest.txt tests this.
				if supported != "en_*_GB" {
					log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
				}
				continue
			}
			ri := regionIntelligibility{
				lang:     b.langIndex(d[0]),
				distance: uint8(distance),
			}
			if d[1] != "*" {
				ri.script = uint8(b.scriptIndex(d[1]))
			}
			switch {
			case d[2] == "*":
				ri.group = 0x80 // not contained in anything
			case strings.HasPrefix(d[2], "$!"):
				ri.group = 0x80
				d[2] = "$" + d[2][len("$!"):]
				fallthrough
			case strings.HasPrefix(d[2], "$"):
				ri.group |= idToIndex[d[2]]
			}
			matchRegion = append(matchRegion, ri)
		default:
			log.Fatalf("not supported: desired=%q; supported=%q", desired, supported)
		}
	}
	sort.SliceStable(matchLang, func(i, j int) bool {
		return matchLang[i].distance < matchLang[j].distance
	})
	b.w.WriteComment(`
		matchLang holds pairs of langIDs of base languages that are typically
		mutually intelligible. Each pair is associated with a confidence and
		whether the intelligibility goes one or both ways.`)
	b.w.WriteVar("matchLang", matchLang)

	b.w.WriteComment(`
		matchScript holds pairs of scriptIDs where readers of one script
		can typically also read the other. Each is associated with a confidence.`)
	sort.SliceStable(matchScript, func(i, j int) bool {
		return matchScript[i].distance < matchScript[j].distance
	})
	b.w.WriteVar("matchScript", matchScript)

	sort.SliceStable(matchRegion, func(i, j int) bool {
		return matchRegion[i].distance < matchRegion[j].distance
	})
	b.w.WriteVar("matchRegion", matchRegion)
}
