package currency

import (
	"flag"
	"strings"
	"testing"
	"time"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
	"golang.org/x/text/unicode/cldr"
)

var draft = flag.String("draft",
	"contributed",
	`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)

func TestTables(t *testing.T) {
	testtext.SkipIfNotLong(t)

	// Read the CLDR zip file.
	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("numbers")
	data, err := d.DecodeZip(r)
	if err != nil {
		t.Fatalf("DecodeZip: %v", err)
	}

	dr, err := cldr.ParseDraft(*draft)
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	for _, lang := range data.Locales() {
		p := message.NewPrinter(language.MustParse(lang))

		ldml := data.RawLDML(lang)
		if ldml.Numbers == nil || ldml.Numbers.Currencies == nil {
			continue
		}
		for _, c := range ldml.Numbers.Currencies.Currency {
			syms := cldr.MakeSlice(&c.Symbol)
			syms.SelectDraft(dr)

			for _, sym := range c.Symbol {
				cur, err := ParseISO(c.Type)
				if err != nil {
					continue
				}
				formatter := Symbol
				switch sym.Alt {
				case "":
				case "narrow":
					formatter = NarrowSymbol
				default:
					continue
				}
				want := sym.Data()
				if got := p.Sprint(formatter(cur)); got != want {
					t.Errorf("%s:%sSymbol(%s) = %s; want %s", lang, strings.Title(sym.Alt), c.Type, got, want)
				}
			}
		}
	}

	for _, reg := range data.Supplemental().CurrencyData.Region {
		i := 0
		for ; regionData[i].Region().String() != reg.Iso3166; i++ {
		}
		it := Query(Historical, NonTender, Region(language.MustParseRegion(reg.Iso3166)))
		for _, cur := range reg.Currency {
			from, _ := time.Parse("2006-01-02", cur.From)
			to, _ := time.Parse("2006-01-02", cur.To)

			it.Next()
			for j, r := range []QueryIter{&iter{regionInfo: &regionData[i]}, it} {
				if got, _ := r.From(); from != got {
					t.Errorf("%d:%s:%s:from: got %v; want %v", j, reg.Iso3166, cur.Iso4217, got, from)
				}
				if got, _ := r.To(); to != got {
					t.Errorf("%d:%s:%s:to: got %v; want %v", j, reg.Iso3166, cur.Iso4217, got, to)
				}
			}
			i++
		}
	}
}
