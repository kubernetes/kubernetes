// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldrtree

import (
	"bytes"
	"flag"
	"io/ioutil"
	"log"
	"math/rand"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/language/compact"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var genOutput = flag.Bool("gen", false, "generate output files")

func TestAliasRegexp(t *testing.T) {
	testCases := []struct {
		alias string
		want  []string
	}{{
		alias: "miscPatterns[@numberSystem='latn']",
		want: []string{
			"miscPatterns[@numberSystem='latn']",
			"miscPatterns",
			"[@numberSystem='latn']",
			"numberSystem",
			"latn",
		},
	}, {
		alias: `calendar[@type='greg-foo']/days/`,
		want: []string{
			"calendar[@type='greg-foo']",
			"calendar",
			"[@type='greg-foo']",
			"type",
			"greg-foo",
		},
	}, {
		alias: "eraAbbr",
		want: []string{
			"eraAbbr",
			"eraAbbr",
			"",
			"",
			"",
		},
	}, {
		// match must be anchored at beginning.
		alias: `../calendar[@type='gregorian']/days/`,
	}}
	for _, tc := range testCases {
		t.Run(tc.alias, func(t *testing.T) {
			got := aliasRe.FindStringSubmatch(tc.alias)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %v; want %v", got, tc.want)
			}
		})
	}
}

func TestBuild(t *testing.T) {
	tree1, _ := loadTestdata(t, "test1")
	tree2, _ := loadTestdata(t, "test2")

	// Constants for second test
	const (
		calendar = iota
		field
	)
	const (
		month = iota
		era
		filler
		cyclicNameSet
	)
	const (
		abbreviated = iota
		narrow
		wide
	)

	testCases := []struct {
		desc      string
		tree      *Tree
		locale    string
		path      []uint16
		isFeature bool
		result    string
	}{{
		desc:   "und/chinese month format wide m1",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 0, month, 0, wide, 1),
		result: "cM01",
	}, {
		desc:   "und/chinese month format wide m12",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 0, month, 0, wide, 12),
		result: "cM12",
	}, {
		desc:   "und/non-existing value",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 0, month, 0, wide, 13),
		result: "",
	}, {
		desc:   "und/dangi:chinese month format wide",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 1, month, 0, wide, 1),
		result: "cM01",
	}, {
		desc:   "und/chinese month format abbreviated:wide",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 0, month, 0, abbreviated, 1),
		result: "cM01",
	}, {
		desc:   "und/chinese month format narrow:wide",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 0, month, 0, narrow, 1),
		result: "cM01",
	}, {
		desc:   "und/gregorian month format wide",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 2, month, 0, wide, 2),
		result: "gM02",
	}, {
		desc:   "und/gregorian month format:stand-alone narrow",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 2, month, 0, narrow, 1),
		result: "1",
	}, {
		desc:   "und/gregorian month stand-alone:format abbreviated",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 2, month, 1, abbreviated, 1),
		result: "gM01",
	}, {
		desc:   "und/gregorian month stand-alone:format wide ",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 2, month, 1, abbreviated, 1),
		result: "gM01",
	}, {
		desc:   "und/dangi:chinese month format narrow:wide ",
		tree:   tree1,
		locale: "und",
		path:   path(calendar, 1, month, 0, narrow, 4),
		result: "cM04",
	}, {
		desc:   "und/field era displayname 0",
		tree:   tree2,
		locale: "und",
		path:   path(field, 0, 0, 0),
		result: "Era",
	}, {
		desc:   "en/field era displayname 0",
		tree:   tree2,
		locale: "en",
		path:   path(field, 0, 0, 0),
		result: "era",
	}, {
		desc:   "und/calendar hebrew format wide 7-leap",
		tree:   tree2,
		locale: "und",
		path:   path(calendar, 7, month, 0, wide, 0),
		result: "Adar II",
	}, {
		desc:   "en-GB:en-001:en:und/calendar hebrew format wide 7-leap",
		tree:   tree2,
		locale: "en-GB",
		path:   path(calendar, 7, month, 0, wide, 0),
		result: "Adar II",
	}, {
		desc:   "und/buddhist month format wide 11",
		tree:   tree2,
		locale: "und",
		path:   path(calendar, 0, month, 0, wide, 12),
		result: "genWideM12",
	}, {
		desc:   "en-GB/gregorian month stand-alone narrow 2",
		tree:   tree2,
		locale: "en-GB",
		path:   path(calendar, 6, month, 1, narrow, 3),
		result: "gbNarrowM3",
	}, {
		desc:   "en-GB/gregorian month format narrow 3/missing in en-GB",
		tree:   tree2,
		locale: "en-GB",
		path:   path(calendar, 6, month, 0, narrow, 4),
		result: "enNarrowM4",
	}, {
		desc:   "en-GB/gregorian month format narrow 3/missing in en and en-GB",
		tree:   tree2,
		locale: "en-GB",
		path:   path(calendar, 6, month, 0, narrow, 7),
		result: "gregNarrowM7",
	}, {
		desc:   "en-GB/gregorian month format narrow 3/missing in en and en-GB",
		tree:   tree2,
		locale: "en-GB",
		path:   path(calendar, 6, month, 0, narrow, 7),
		result: "gregNarrowM7",
	}, {
		desc:      "en-GB/gregorian era narrow",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(calendar, 6, era, abbreviated, 0, 1),
		isFeature: true,
		result:    "AD",
	}, {
		desc:      "en-GB/gregorian era narrow",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(calendar, 6, era, narrow, 0, 0),
		isFeature: true,
		result:    "BC",
	}, {
		desc:      "en-GB/gregorian era narrow",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(calendar, 6, era, wide, 1, 0),
		isFeature: true,
		result:    "Before Common Era",
	}, {
		desc:      "en-GB/dangi:chinese cyclicName, months, format, narrow:abbreviated 2",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(calendar, 1, cyclicNameSet, 3, 0, 1, 2),
		isFeature: true,
		result:    "year2",
	}, {
		desc:   "en-GB/field era-narrow ",
		tree:   tree2,
		locale: "en-GB",
		path:   path(field, 2, 0, 0),
		result: "era",
	}, {
		desc:      "en-GB/field month-narrow relativeTime future one",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(field, 5, 2, 0, 1),
		isFeature: true,
		result:    "001NarrowFutMOne",
	}, {
		// Don't fall back to the one of "en".
		desc:      "en-GB/field month-short relativeTime past one:other",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(field, 4, 2, 1, 1),
		isFeature: true,
		result:    "001ShortPastMOther",
	}, {
		desc:      "en-GB/field month relativeTime future two:other",
		tree:      tree2,
		locale:    "en-GB",
		path:      path(field, 3, 2, 0, 2),
		isFeature: true,
		result:    "enFutMOther",
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			tag, _ := compact.RegionalID(compact.Tag(language.MustParse(tc.locale)))
			s := tc.tree.lookup(tag, tc.isFeature, tc.path...)
			if s != tc.result {
				t.Errorf("got %q; want %q", s, tc.result)
			}
		})
	}
}

func path(e ...uint16) []uint16 { return e }

func TestGen(t *testing.T) {
	testCases := []string{"test1", "test2"}
	for _, tc := range testCases {
		t.Run(tc, func(t *testing.T) {
			_, got := loadTestdata(t, tc)

			// Remove sizes that may vary per architecture.
			re := regexp.MustCompile("// Size: [0-9]*")
			got = re.ReplaceAllLiteral(got, []byte("// Size: xxxx"))
			re = regexp.MustCompile("// Total table size [0-9]*")
			got = re.ReplaceAllLiteral(got, []byte("// Total table size: xxxx"))

			file := filepath.Join("testdata", tc, "output.go")
			if *genOutput {
				ioutil.WriteFile(file, got, 0700)
				t.SkipNow()
			}

			b, err := ioutil.ReadFile(file)
			if err != nil {
				t.Fatalf("failed to open file: %v", err)
			}
			if want := string(b); string(got) != want {
				t.Log(string(got))
				t.Errorf("files differ")
			}
		})
	}
}

func loadTestdata(t *testing.T, test string) (tree *Tree, file []byte) {
	b := New("test")

	var d cldr.Decoder

	data, err := d.DecodePath(filepath.Join("testdata", test))
	if err != nil {
		t.Fatalf("error decoding testdata: %v", err)
	}

	context := Enum("context")
	widthMap := func(s string) string {
		// Align era with width values.
		if r, ok := map[string]string{
			"eraAbbr":   "abbreviated",
			"eraNarrow": "narrow",
			"eraNames":  "wide",
		}[s]; ok {
			s = r
		}
		return "w" + strings.Title(s)
	}
	width := EnumFunc("width", widthMap, "abbreviated", "narrow", "wide")
	month := Enum("month", "leap7")
	relative := EnumFunc("relative", func(s string) string {
		x, err := strconv.ParseInt(s, 10, 8)
		if err != nil {
			log.Fatal("Invalid number:", err)
		}
		return []string{
			"before1",
			"current",
			"after1",
		}[x+1]
	})
	cycleType := EnumFunc("cycleType", func(s string) string {
		return "cyc" + strings.Title(s)
	})
	r := rand.New(rand.NewSource(0))

	for _, loc := range data.Locales() {
		ldml := data.RawLDML(loc)
		x := b.Locale(language.Make(loc))

		if x := x.Index(ldml.Dates.Calendars); x != nil {
			for _, cal := range ldml.Dates.Calendars.Calendar {
				x := x.IndexFromType(cal)
				if x := x.Index(cal.Months); x != nil {
					for _, mc := range cal.Months.MonthContext {
						x := x.IndexFromType(mc, context)
						for _, mw := range mc.MonthWidth {
							x := x.IndexFromType(mw, width)
							for _, m := range mw.Month {
								x.SetValue(m.Yeartype+m.Type, m, month)
							}
						}
					}
				}
				if x := x.Index(cal.CyclicNameSets); x != nil {
					for _, cns := range cal.CyclicNameSets.CyclicNameSet {
						x := x.IndexFromType(cns, cycleType)
						for _, cc := range cns.CyclicNameContext {
							x := x.IndexFromType(cc, context)
							for _, cw := range cc.CyclicNameWidth {
								x := x.IndexFromType(cw, width)
								for _, c := range cw.CyclicName {
									x.SetValue(c.Type, c)
								}
							}
						}
					}
				}
				if x := x.Index(cal.Eras); x != nil {
					opts := []Option{width, SharedType()}
					if x := x.Index(cal.Eras.EraNames, opts...); x != nil {
						for _, e := range cal.Eras.EraNames.Era {
							x.IndexFromAlt(e).SetValue(e.Type, e)
						}
					}
					if x := x.Index(cal.Eras.EraAbbr, opts...); x != nil {
						for _, e := range cal.Eras.EraAbbr.Era {
							x.IndexFromAlt(e).SetValue(e.Type, e)
						}
					}
					if x := x.Index(cal.Eras.EraNarrow, opts...); x != nil {
						for _, e := range cal.Eras.EraNarrow.Era {
							x.IndexFromAlt(e).SetValue(e.Type, e)
						}
					}
				}
				{
					// Ensure having more than 2 buckets.
					f := x.IndexWithName("filler")
					b := make([]byte, maxStrlen)
					opt := &options{parent: x}
					r.Read(b)
					f.setValue("0", string(b), opt)
				}
			}
		}
		if x := x.Index(ldml.Dates.Fields); x != nil {
			for _, f := range ldml.Dates.Fields.Field {
				x := x.IndexFromType(f)
				for _, d := range f.DisplayName {
					x.Index(d).SetValue("", d)
				}
				for _, r := range f.Relative {
					x.Index(r).SetValue(r.Type, r, relative)
				}
				for _, rt := range f.RelativeTime {
					x := x.Index(rt).IndexFromType(rt)
					for _, p := range rt.RelativeTimePattern {
						x.SetValue(p.Count, p)
					}
				}
				for _, rp := range f.RelativePeriod {
					x.Index(rp).SetValue("", rp)
				}
			}
		}
	}

	tree, err = build(b)
	if err != nil {
		t.Fatal("error building tree:", err)
	}
	w := gen.NewCodeWriter()
	generate(b, tree, w)
	generateTestData(b, w)
	buf := &bytes.Buffer{}
	if _, err = w.WriteGo(buf, "test", ""); err != nil {
		t.Log(buf.String())
		t.Fatal("error generating code:", err)
	}
	return tree, buf.Bytes()
}
