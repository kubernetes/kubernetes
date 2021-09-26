// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package date

import (
	"strconv"
	"strings"
	"testing"

	"golang.org/x/text/internal/cldrtree"
	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/language/compact"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/cldr"
)

func TestTables(t *testing.T) {
	testtext.SkipIfNotLong(t)

	r := gen.OpenCLDRCoreZip()
	defer r.Close()

	d := &cldr.Decoder{}
	d.SetDirFilter("supplemental", "main")
	d.SetSectionFilter("dates")
	data, err := d.DecodeZip(r)
	if err != nil {
		t.Fatalf("DecodeZip: %v", err)
	}

	count := 0
	for _, lang := range data.Locales() {
		ldml := data.RawLDML(lang)
		if ldml.Dates == nil {
			continue
		}
		tag, _ := compact.RegionalID(compact.Tag(language.MustParse(lang)))

		test := func(want cldrtree.Element, path ...string) {
			if count > 30 {
				return
			}
			t.Run(lang+"/"+strings.Join(path, "/"), func(t *testing.T) {
				p := make([]uint16, len(path))
				for i, s := range path {
					if v, err := strconv.Atoi(s); err == nil {
						p[i] = uint16(v)
					} else if v, ok := enumMap[s]; ok {
						p[i] = v
					} else {
						count++
						t.Fatalf("Unknown key %q", s)
					}
				}
				wantStr := want.GetCommon().Data()
				if got := tree.Lookup(tag, p...); got != wantStr {
					count++
					t.Errorf("got %q; want %q", got, wantStr)
				}
			})
		}

		width := func(s string) string { return "width" + strings.Title(s) }

		if ldml.Dates.Calendars != nil {
			for _, cal := range ldml.Dates.Calendars.Calendar {
				if cal.Months != nil {
					for _, mc := range cal.Months.MonthContext {
						for _, mw := range mc.MonthWidth {
							for _, m := range mw.Month {
								test(m, "calendars", cal.Type, "months", mc.Type, width(mw.Type), m.Yeartype+m.Type)
							}
						}
					}
				}
				if cal.MonthPatterns != nil {
					for _, mc := range cal.MonthPatterns.MonthPatternContext {
						for _, mw := range mc.MonthPatternWidth {
							for _, m := range mw.MonthPattern {
								test(m, "calendars", cal.Type, "monthPatterns", mc.Type, width(mw.Type))
							}
						}
					}
				}
				if cal.CyclicNameSets != nil {
					for _, cns := range cal.CyclicNameSets.CyclicNameSet {
						for _, cc := range cns.CyclicNameContext {
							for _, cw := range cc.CyclicNameWidth {
								for _, c := range cw.CyclicName {
									test(c, "calendars", cal.Type, "cyclicNameSets", cns.Type+"CycleType", cc.Type, width(cw.Type), c.Type)

								}
							}
						}
					}
				}
				if cal.Days != nil {
					for _, dc := range cal.Days.DayContext {
						for _, dw := range dc.DayWidth {
							for _, d := range dw.Day {
								test(d, "calendars", cal.Type, "days", dc.Type, width(dw.Type), d.Type)
							}
						}
					}
				}
				if cal.Quarters != nil {
					for _, qc := range cal.Quarters.QuarterContext {
						for _, qw := range qc.QuarterWidth {
							for _, q := range qw.Quarter {
								test(q, "calendars", cal.Type, "quarters", qc.Type, width(qw.Type), q.Type)
							}
						}
					}
				}
				if cal.DayPeriods != nil {
					for _, dc := range cal.DayPeriods.DayPeriodContext {
						for _, dw := range dc.DayPeriodWidth {
							for _, d := range dw.DayPeriod {
								test(d, "calendars", cal.Type, "dayPeriods", dc.Type, width(dw.Type), d.Type, d.Alt)
							}
						}
					}
				}
				if cal.Eras != nil {
					if cal.Eras.EraNames != nil {
						for _, e := range cal.Eras.EraNames.Era {
							test(e, "calendars", cal.Type, "eras", "widthWide", e.Alt, e.Type)
						}
					}
					if cal.Eras.EraAbbr != nil {
						for _, e := range cal.Eras.EraAbbr.Era {
							test(e, "calendars", cal.Type, "eras", "widthAbbreviated", e.Alt, e.Type)
						}
					}
					if cal.Eras.EraNarrow != nil {
						for _, e := range cal.Eras.EraNarrow.Era {
							test(e, "calendars", cal.Type, "eras", "widthNarrow", e.Alt, e.Type)
						}
					}
				}
				if cal.DateFormats != nil {
					for _, dfl := range cal.DateFormats.DateFormatLength {
						for _, df := range dfl.DateFormat {
							for _, p := range df.Pattern {
								test(p, "calendars", cal.Type, "dateFormats", dfl.Type, p.Alt)
							}
						}
					}
				}
				if cal.TimeFormats != nil {
					for _, tfl := range cal.TimeFormats.TimeFormatLength {
						for _, tf := range tfl.TimeFormat {
							for _, p := range tf.Pattern {
								test(p, "calendars", cal.Type, "timeFormats", tfl.Type, p.Alt)
							}
						}
					}
				}
				if cal.DateTimeFormats != nil {
					for _, dtfl := range cal.DateTimeFormats.DateTimeFormatLength {
						for _, dtf := range dtfl.DateTimeFormat {
							for _, p := range dtf.Pattern {
								test(p, "calendars", cal.Type, "dateTimeFormats", dtfl.Type, p.Alt)
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
		if ldml.Dates.Fields != nil {
			for _, f := range ldml.Dates.Fields.Field {
				field := f.Type + "Field"
				for _, d := range f.DisplayName {
					test(d, "fields", field, "displayName", d.Alt)
				}
				for _, r := range f.Relative {
					i, _ := strconv.Atoi(r.Type)
					v := []string{"before2", "before1", "current", "after1", "after2", "after3"}[i+2]
					test(r, "fields", field, "relative", v)
				}
				for _, rt := range f.RelativeTime {
					for _, p := range rt.RelativeTimePattern {
						test(p, "fields", field, "relativeTime", rt.Type, p.Count)
					}
				}
				for _, rp := range f.RelativePeriod {
					test(rp, "fields", field, "relativePeriod", rp.Alt)
				}
			}
		}
		if ldml.Dates.TimeZoneNames != nil {
			for _, h := range ldml.Dates.TimeZoneNames.HourFormat {
				test(h, "timeZoneNames", "zoneFormat", h.Element())
			}
			for _, g := range ldml.Dates.TimeZoneNames.GmtFormat {
				test(g, "timeZoneNames", "zoneFormat", g.Element())
			}
			for _, g := range ldml.Dates.TimeZoneNames.GmtZeroFormat {
				test(g, "timeZoneNames", "zoneFormat", g.Element())
			}
			for _, r := range ldml.Dates.TimeZoneNames.RegionFormat {
				s := r.Type
				if s == "" {
					s = "generic"
				}
				test(r, "timeZoneNames", "regionFormat", s+"Time")
			}

			testZone := func(zoneType, zoneWidth, zone string, a ...[]*cldr.Common) {
				for _, e := range a {
					for _, n := range e {
						test(n, "timeZoneNames", zoneType, zoneWidth, n.Element()+"Time", zone)
					}
				}
			}
			for _, z := range ldml.Dates.TimeZoneNames.Zone {
				for _, l := range z.Long {
					testZone("zone", l.Element(), z.Type, l.Generic, l.Standard, l.Daylight)
				}
				for _, l := range z.Short {
					testZone("zone", l.Element(), z.Type, l.Generic, l.Standard, l.Daylight)
				}
			}
			for _, z := range ldml.Dates.TimeZoneNames.Metazone {
				for _, l := range z.Long {
					testZone("metaZone", l.Element(), z.Type, l.Generic, l.Standard, l.Daylight)
				}
				for _, l := range z.Short {
					testZone("metaZone", l.Element(), z.Type, l.Generic, l.Standard, l.Daylight)
				}
			}
		}
	}
}
