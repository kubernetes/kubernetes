// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"flag"
	"log"
	"reflect"
	"testing"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var draft = flag.String("draft",
	"contributed",
	`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)

func TestNumberSystems(t *testing.T) {
	testtext.SkipIfNotLong(t)

	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental")
	d.SetSectionFilter("numberingSystem")
	data, err := d.DecodeZip(r)
	if err != nil {
		t.Fatalf("DecodeZip: %v", err)
	}

	for _, ns := range data.Supplemental().NumberingSystems.NumberingSystem {
		n := systemMap[ns.Id]
		if int(n) >= len(numSysData) {
			continue
		}
		info := InfoFromLangID(0, ns.Id)
		val := '0'
		for _, rWant := range ns.Digits {
			if rGot := info.Digit(val); rGot != rWant {
				t.Errorf("%s:%d: got %U; want %U", ns.Id, val, rGot, rWant)
			}
			val++
		}
	}
}

func TestSymbols(t *testing.T) {
	testtext.SkipIfNotLong(t)

	draft, err := cldr.ParseDraft(*draft)
	if err != nil {
		log.Fatalf("invalid draft level: %v", err)
	}

	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("main")
	d.SetSectionFilter("numbers")
	data, err := d.DecodeZip(r)
	if err != nil {
		t.Fatalf("DecodeZip: %v", err)
	}

	for _, lang := range data.Locales() {
		ldml := data.RawLDML(lang)
		if ldml.Numbers == nil {
			continue
		}
		langIndex, ok := language.CompactIndex(language.MustParse(lang))
		if !ok {
			t.Fatalf("No compact index for language %s", lang)
		}

		syms := cldr.MakeSlice(&ldml.Numbers.Symbols)
		syms.SelectDraft(draft)

		for _, sym := range ldml.Numbers.Symbols {
			if sym.NumberSystem == "" {
				continue
			}
			testCases := []struct {
				name string
				st   SymbolType
				x    interface{}
			}{
				{"Decimal", SymDecimal, sym.Decimal},
				{"Group", SymGroup, sym.Group},
				{"List", SymList, sym.List},
				{"PercentSign", SymPercentSign, sym.PercentSign},
				{"PlusSign", SymPlusSign, sym.PlusSign},
				{"MinusSign", SymMinusSign, sym.MinusSign},
				{"Exponential", SymExponential, sym.Exponential},
				{"SuperscriptingExponent", SymSuperscriptingExponent, sym.SuperscriptingExponent},
				{"PerMille", SymPerMille, sym.PerMille},
				{"Infinity", SymInfinity, sym.Infinity},
				{"NaN", SymNan, sym.Nan},
				{"TimeSeparator", SymTimeSeparator, sym.TimeSeparator},
			}
			info := InfoFromLangID(langIndex, sym.NumberSystem)
			for _, tc := range testCases {
				// Extract the wanted value.
				v := reflect.ValueOf(tc.x)
				if v.Len() == 0 {
					return
				}
				if v.Len() > 1 {
					t.Fatalf("Multiple values of %q within single symbol not supported.", tc.name)
				}
				want := v.Index(0).MethodByName("Data").Call(nil)[0].String()
				got := info.Symbol(tc.st)
				if got != want {
					t.Errorf("%s:%s:%s: got %q; want %q", lang, sym.NumberSystem, tc.name, got, want)
				}
			}
		}
	}
}
