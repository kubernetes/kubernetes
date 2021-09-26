// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"flag"
	"log"
	"strconv"
	"strings"

	"golang.org/x/text/internal/cldrtree"
	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

var (
	draft = flag.String("draft",
		"contributed",
		`Minimal draft requirements (approved, contributed, provisional, unconfirmed).`)
)

// TODO:
// - Compile format patterns.
// - Compress the large amount of redundancy in metazones.
// - Split trees (with shared buckets) with data that is enough for default
//   formatting of Go Time values and tables that are needed for larger
//   variants.
// - zone to metaZone mappings (in supplemental)
// - Add more enum values and also some key maps for some of the elements.

func main() {
	gen.Init()

	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("dates")
	data, err := d.DecodeZip(r)
	if err != nil {
		log.Fatalf("DecodeZip: %v", err)
	}

	dates := cldrtree.New("dates")
	buildCLDRTree(data, dates)

	w := gen.NewCodeWriter()
	if err := dates.Gen(w); err != nil {
		log.Fatal(err)
	}
	gen.WriteCLDRVersion(w)
	w.WriteGoFile("tables.go", "date")

	w = gen.NewCodeWriter()
	if err := dates.GenTestData(w); err != nil {
		log.Fatal(err)
	}
	w.WriteGoFile("data_test.go", "date")
}

func buildCLDRTree(data *cldr.CLDR, dates *cldrtree.Builder) {
	context := cldrtree.Enum("context")
	widthMap := func(s string) string {
		// Align era with width values.
		if r, ok := map[string]string{
			"eraAbbr":   "abbreviated",
			"eraNarrow": "narrow",
			"eraNames":  "wide",
		}[s]; ok {
			s = r
		}
		// Prefix width to disambiguate with some overlapping length values.
		return "width" + strings.Title(s)
	}
	width := cldrtree.EnumFunc("width", widthMap, "abbreviated", "narrow", "wide")
	length := cldrtree.Enum("length", "short", "long")
	month := cldrtree.Enum("month", "leap7")
	relTime := cldrtree.EnumFunc("relTime", func(s string) string {
		x, err := strconv.ParseInt(s, 10, 8)
		if err != nil {
			log.Fatal("Invalid number:", err)
		}
		return []string{
			"before2",
			"before1",
			"current",
			"after1",
			"after2",
			"after3",
		}[x+2]
	})
	// Disambiguate keys like 'months' and 'sun'.
	cycleType := cldrtree.EnumFunc("cycleType", func(s string) string {
		return s + "CycleType"
	})
	field := cldrtree.EnumFunc("field", func(s string) string {
		return s + "Field"
	})
	timeType := cldrtree.EnumFunc("timeType", func(s string) string {
		if s == "" {
			return "genericTime"
		}
		return s + "Time"
	}, "generic")

	zoneType := []cldrtree.Option{cldrtree.SharedType(), timeType}
	metaZoneType := []cldrtree.Option{cldrtree.SharedType(), timeType}

	for _, lang := range data.Locales() {
		tag := language.Make(lang)
		ldml := data.RawLDML(lang)
		if ldml.Dates == nil {
			continue
		}
		x := dates.Locale(tag)
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
				if x := x.Index(cal.MonthPatterns); x != nil {
					for _, mc := range cal.MonthPatterns.MonthPatternContext {
						x := x.IndexFromType(mc, context)
						for _, mw := range mc.MonthPatternWidth {
							// Value is always leap, so no need to create a
							// subindex.
							for _, m := range mw.MonthPattern {
								x.SetValue(mw.Type, m, width)
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
				if x := x.Index(cal.Days); x != nil {
					for _, dc := range cal.Days.DayContext {
						x := x.IndexFromType(dc, context)
						for _, dw := range dc.DayWidth {
							x := x.IndexFromType(dw, width)
							for _, d := range dw.Day {
								x.SetValue(d.Type, d)
							}
						}
					}
				}
				if x := x.Index(cal.Quarters); x != nil {
					for _, qc := range cal.Quarters.QuarterContext {
						x := x.IndexFromType(qc, context)
						for _, qw := range qc.QuarterWidth {
							x := x.IndexFromType(qw, width)
							for _, q := range qw.Quarter {
								x.SetValue(q.Type, q)
							}
						}
					}
				}
				if x := x.Index(cal.DayPeriods); x != nil {
					for _, dc := range cal.DayPeriods.DayPeriodContext {
						x := x.IndexFromType(dc, context)
						for _, dw := range dc.DayPeriodWidth {
							x := x.IndexFromType(dw, width)
							for _, d := range dw.DayPeriod {
								x.IndexFromType(d).SetValue(d.Alt, d)
							}
						}
					}
				}
				if x := x.Index(cal.Eras); x != nil {
					opts := []cldrtree.Option{width, cldrtree.SharedType()}
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
				if x := x.Index(cal.DateFormats); x != nil {
					for _, dfl := range cal.DateFormats.DateFormatLength {
						x := x.IndexFromType(dfl, length)
						for _, df := range dfl.DateFormat {
							for _, p := range df.Pattern {
								x.SetValue(p.Alt, p)
							}
						}
					}
				}
				if x := x.Index(cal.TimeFormats); x != nil {
					for _, tfl := range cal.TimeFormats.TimeFormatLength {
						x := x.IndexFromType(tfl, length)
						for _, tf := range tfl.TimeFormat {
							for _, p := range tf.Pattern {
								x.SetValue(p.Alt, p)
							}
						}
					}
				}
				if x := x.Index(cal.DateTimeFormats); x != nil {
					for _, dtfl := range cal.DateTimeFormats.DateTimeFormatLength {
						x := x.IndexFromType(dtfl, length)
						for _, dtf := range dtfl.DateTimeFormat {
							for _, p := range dtf.Pattern {
								x.SetValue(p.Alt, p)
							}
						}
					}
					// TODO:
					// - appendItems
					// - intervalFormats
				}
			}
		}
		// TODO: this is a lot of data and is probably relatively little used.
		// Store this somewhere else.
		if x := x.Index(ldml.Dates.Fields); x != nil {
			for _, f := range ldml.Dates.Fields.Field {
				x := x.IndexFromType(f, field)
				for _, d := range f.DisplayName {
					x.Index(d).SetValue(d.Alt, d)
				}
				for _, r := range f.Relative {
					x.Index(r).SetValue(r.Type, r, relTime)
				}
				for _, rt := range f.RelativeTime {
					x := x.Index(rt).IndexFromType(rt)
					for _, p := range rt.RelativeTimePattern {
						x.SetValue(p.Count, p)
					}
				}
				for _, rp := range f.RelativePeriod {
					x.Index(rp).SetValue(rp.Alt, rp)
				}
			}
		}
		if x := x.Index(ldml.Dates.TimeZoneNames); x != nil {
			format := x.IndexWithName("zoneFormat")
			for _, h := range ldml.Dates.TimeZoneNames.HourFormat {
				format.SetValue(h.Element(), h)
			}
			for _, g := range ldml.Dates.TimeZoneNames.GmtFormat {
				format.SetValue(g.Element(), g)
			}
			for _, g := range ldml.Dates.TimeZoneNames.GmtZeroFormat {
				format.SetValue(g.Element(), g)
			}
			for _, r := range ldml.Dates.TimeZoneNames.RegionFormat {
				x.Index(r).SetValue(r.Type, r, timeType)
			}

			set := func(x *cldrtree.Index, e []*cldr.Common, zone string) {
				for _, n := range e {
					x.Index(n, zoneType...).SetValue(zone, n)
				}
			}
			zoneWidth := []cldrtree.Option{length, cldrtree.SharedType()}
			zs := x.IndexWithName("zone")
			for _, z := range ldml.Dates.TimeZoneNames.Zone {
				for _, l := range z.Long {
					x := zs.Index(l, zoneWidth...)
					set(x, l.Generic, z.Type)
					set(x, l.Standard, z.Type)
					set(x, l.Daylight, z.Type)
				}
				for _, s := range z.Short {
					x := zs.Index(s, zoneWidth...)
					set(x, s.Generic, z.Type)
					set(x, s.Standard, z.Type)
					set(x, s.Daylight, z.Type)
				}
			}
			set = func(x *cldrtree.Index, e []*cldr.Common, zone string) {
				for _, n := range e {
					x.Index(n, metaZoneType...).SetValue(zone, n)
				}
			}
			zoneWidth = []cldrtree.Option{length, cldrtree.SharedType()}
			zs = x.IndexWithName("metaZone")
			for _, z := range ldml.Dates.TimeZoneNames.Metazone {
				for _, l := range z.Long {
					x := zs.Index(l, zoneWidth...)
					set(x, l.Generic, z.Type)
					set(x, l.Standard, z.Type)
					set(x, l.Daylight, z.Type)
				}
				for _, s := range z.Short {
					x := zs.Index(s, zoneWidth...)
					set(x, s.Generic, z.Type)
					set(x, s.Standard, z.Type)
					set(x, s.Daylight, z.Type)
				}
			}
		}
	}
}
