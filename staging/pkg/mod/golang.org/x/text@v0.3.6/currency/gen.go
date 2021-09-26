// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// Generator for currency-related data.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/text/internal/language/compact"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/tag"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var (
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")
	outputFile = flag.String("output", "tables.go", "output file")

	draft = flag.String("draft",
		"contributed",
		`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)
)

func main() {
	gen.Init()

	gen.Repackage("gen_common.go", "common.go", "currency")

	// Read the CLDR zip file.
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("numbers")
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	w := gen.NewCodeWriter()
	defer w.WriteGoFile(*outputFile, "currency")

	fmt.Fprintln(w, `import "golang.org/x/text/internal/tag"`)

	gen.WriteCLDRVersion(w)
	b := &builder{}
	b.genCurrencies(w, data.Supplemental())
	b.genSymbols(w, data)
}

var constants = []string{
	// Undefined and testing.
	"XXX", "XTS",
	// G11 currencies https://en.wikipedia.org/wiki/G10_currencies.
	"USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK",
	// Precious metals.
	"XAG", "XAU", "XPT", "XPD",

	// Additional common currencies as defined by CLDR.
	"BRL", "CNY", "INR", "RUB", "HKD", "IDR", "KRW", "MXN", "PLN", "SAR",
	"THB", "TRY", "TWD", "ZAR",
}

type builder struct {
	currencies    tag.Index
	numCurrencies int
}

func (b *builder) genCurrencies(w *gen.CodeWriter, data *cldr.SupplementalData) {
	// 3-letter ISO currency codes
	// Start with dummy to let index start at 1.
	currencies := []string{"\x00\x00\x00\x00"}

	// currency codes
	for _, reg := range data.CurrencyData.Region {
		for _, cur := range reg.Currency {
			currencies = append(currencies, cur.Iso4217)
		}
	}
	// Not included in the list for some reasons:
	currencies = append(currencies, "MVP")

	sort.Strings(currencies)
	// Unique the elements.
	k := 0
	for i := 1; i < len(currencies); i++ {
		if currencies[k] != currencies[i] {
			currencies[k+1] = currencies[i]
			k++
		}
	}
	currencies = currencies[:k+1]

	// Close with dummy for simpler and faster searching.
	currencies = append(currencies, "\xff\xff\xff\xff")

	// Write currency values.
	fmt.Fprintln(w, "const (")
	for _, c := range constants {
		index := sort.SearchStrings(currencies, c)
		fmt.Fprintf(w, "\t%s = %d\n", strings.ToLower(c), index)
	}
	fmt.Fprint(w, ")")

	// Compute currency-related data that we merge into the table.
	for _, info := range data.CurrencyData.Fractions[0].Info {
		if info.Iso4217 == "DEFAULT" {
			continue
		}
		standard := getRoundingIndex(info.Digits, info.Rounding, 0)
		cash := getRoundingIndex(info.CashDigits, info.CashRounding, standard)

		index := sort.SearchStrings(currencies, info.Iso4217)
		currencies[index] += mkCurrencyInfo(standard, cash)
	}

	// Set default values for entries that weren't touched.
	for i, c := range currencies {
		if len(c) == 3 {
			currencies[i] += mkCurrencyInfo(0, 0)
		}
	}

	b.currencies = tag.Index(strings.Join(currencies, ""))
	w.WriteComment(`
	currency holds an alphabetically sorted list of canonical 3-letter currency
	identifiers. Each identifier is followed by a byte of type currencyInfo,
	defined in gen_common.go.`)
	w.WriteConst("currency", b.currencies)

	// Hack alert: gofmt indents a trailing comment after an indented string.
	// Ensure that the next thing written is not a comment.
	b.numCurrencies = (len(b.currencies) / 4) - 2
	w.WriteConst("numCurrencies", b.numCurrencies)

	// Create a table that maps regions to currencies.
	regionToCurrency := []toCurrency{}

	for _, reg := range data.CurrencyData.Region {
		if len(reg.Iso3166) != 2 {
			log.Fatalf("Unexpected group %q in region data", reg.Iso3166)
		}
		if len(reg.Currency) == 0 {
			continue
		}
		cur := reg.Currency[0]
		if cur.To != "" || cur.Tender == "false" {
			continue
		}
		regionToCurrency = append(regionToCurrency, toCurrency{
			region: regionToCode(language.MustParseRegion(reg.Iso3166)),
			code:   uint16(b.currencies.Index([]byte(cur.Iso4217))),
		})
	}
	sort.Sort(byRegion(regionToCurrency))

	w.WriteType(toCurrency{})
	w.WriteVar("regionToCurrency", regionToCurrency)

	// Create a table that maps regions to currencies.
	regionData := []regionInfo{}

	for _, reg := range data.CurrencyData.Region {
		if len(reg.Iso3166) != 2 {
			log.Fatalf("Unexpected group %q in region data", reg.Iso3166)
		}
		for _, cur := range reg.Currency {
			from, _ := time.Parse("2006-01-02", cur.From)
			to, _ := time.Parse("2006-01-02", cur.To)
			code := uint16(b.currencies.Index([]byte(cur.Iso4217)))
			if cur.Tender == "false" {
				code |= nonTenderBit
			}
			regionData = append(regionData, regionInfo{
				region: regionToCode(language.MustParseRegion(reg.Iso3166)),
				code:   code,
				from:   toDate(from),
				to:     toDate(to),
			})
		}
	}
	sort.Stable(byRegionCode(regionData))

	w.WriteType(regionInfo{})
	w.WriteVar("regionData", regionData)
}

type regionInfo struct {
	region uint16
	code   uint16 // 0x8000 not legal tender
	from   uint32
	to     uint32
}

type byRegionCode []regionInfo

func (a byRegionCode) Len() int           { return len(a) }
func (a byRegionCode) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byRegionCode) Less(i, j int) bool { return a[i].region < a[j].region }

type toCurrency struct {
	region uint16
	code   uint16
}

type byRegion []toCurrency

func (a byRegion) Len() int           { return len(a) }
func (a byRegion) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byRegion) Less(i, j int) bool { return a[i].region < a[j].region }

func mkCurrencyInfo(standard, cash int) string {
	return string([]byte{byte(cash<<cashShift | standard)})
}

func getRoundingIndex(digits, rounding string, defIndex int) int {
	round := roundings[defIndex] // default

	if digits != "" {
		round.scale = parseUint8(digits)
	}
	if rounding != "" && rounding != "0" { // 0 means 1 here in CLDR
		round.increment = parseUint8(rounding)
	}

	// Will panic if the entry doesn't exist:
	for i, r := range roundings {
		if r == round {
			return i
		}
	}
	log.Fatalf("Rounding entry %#v does not exist.", round)
	panic("unreachable")
}

// genSymbols generates the symbols used for currencies. Most symbols are
// defined in root and there is only very small variation per language.
// The following rules apply:
// - A symbol can be requested as normal or narrow.
// - If a symbol is not defined for a currency, it defaults to its ISO code.
func (b *builder) genSymbols(w *gen.CodeWriter, data *cldr.CLDR) {
	d, err := cldr.ParseDraft(*draft)
	if err != nil {
		log.Fatalf("filter: %v", err)
	}

	const (
		normal = iota
		narrow
		numTypes
	)
	// language -> currency -> type ->  symbol
	var symbols [compact.NumCompactTags][][numTypes]*string

	// Collect symbol information per language.
	for _, lang := range data.Locales() {
		ldml := data.RawLDML(lang)
		if ldml.Numbers == nil || ldml.Numbers.Currencies == nil {
			continue
		}

		langIndex, ok := compact.LanguageID(compact.Tag(language.MustParse(lang)))
		if !ok {
			log.Fatalf("No compact index for language %s", lang)
		}

		symbols[langIndex] = make([][numTypes]*string, b.numCurrencies+1)

		for _, c := range ldml.Numbers.Currencies.Currency {
			syms := cldr.MakeSlice(&c.Symbol)
			syms.SelectDraft(d)

			for _, sym := range c.Symbol {
				v := sym.Data()
				if v == c.Type {
					// We define "" to mean the ISO symbol.
					v = ""
				}
				cur := b.currencies.Index([]byte(c.Type))
				// XXX gets reassigned to 0 in the package's code.
				if c.Type == "XXX" {
					cur = 0
				}
				if cur == -1 {
					fmt.Println("Unsupported:", c.Type)
					continue
				}

				switch sym.Alt {
				case "":
					symbols[langIndex][cur][normal] = &v
				case "narrow":
					symbols[langIndex][cur][narrow] = &v
				}
			}
		}
	}

	// Remove values identical to the parent.
	for langIndex, data := range symbols {
		for curIndex, curs := range data {
			for typ, sym := range curs {
				if sym == nil {
					continue
				}
				for p := compact.ID(langIndex); p != 0; {
					p = p.Parent()
					x := symbols[p]
					if x == nil {
						continue
					}
					if v := x[curIndex][typ]; v != nil || p == 0 {
						// Value is equal to the default value root value is undefined.
						parentSym := ""
						if v != nil {
							parentSym = *v
						}
						if parentSym == *sym {
							// Value is the same as parent.
							data[curIndex][typ] = nil
						}
						break
					}
				}
			}
		}
	}

	// Create symbol index.
	symbolData := []byte{0}
	symbolLookup := map[string]uint16{"": 0} // 0 means default, so block that value.
	for _, data := range symbols {
		for _, curs := range data {
			for _, sym := range curs {
				if sym == nil {
					continue
				}
				if _, ok := symbolLookup[*sym]; !ok {
					symbolLookup[*sym] = uint16(len(symbolData))
					symbolData = append(symbolData, byte(len(*sym)))
					symbolData = append(symbolData, *sym...)
				}
			}
		}
	}
	w.WriteComment(`
	symbols holds symbol data of the form <n> <str>, where n is the length of
	the symbol string str.`)
	w.WriteConst("symbols", string(symbolData))

	// Create index from language to currency lookup to symbol.
	type curToIndex struct{ cur, idx uint16 }
	w.WriteType(curToIndex{})

	prefix := []string{"normal", "narrow"}
	// Create data for regular and narrow symbol data.
	for typ := normal; typ <= narrow; typ++ {

		indexes := []curToIndex{} // maps currency to symbol index
		languages := []uint16{}

		for _, data := range symbols {
			languages = append(languages, uint16(len(indexes)))
			for curIndex, curs := range data {

				if sym := curs[typ]; sym != nil {
					indexes = append(indexes, curToIndex{uint16(curIndex), symbolLookup[*sym]})
				}
			}
		}
		languages = append(languages, uint16(len(indexes)))

		w.WriteVar(prefix[typ]+"LangIndex", languages)
		w.WriteVar(prefix[typ]+"SymIndex", indexes)
	}
}
func parseUint8(str string) uint8 {
	x, err := strconv.ParseUint(str, 10, 8)
	if err != nil {
		// Show line number of where this function was called.
		log.New(os.Stderr, "", log.Lshortfile).Output(2, err.Error())
		os.Exit(1)
	}
	return uint8(x)
}
