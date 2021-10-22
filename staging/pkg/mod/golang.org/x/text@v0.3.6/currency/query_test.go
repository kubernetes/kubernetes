// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package currency

import (
	"testing"
	"time"

	"golang.org/x/text/language"
)

func TestQuery(t *testing.T) {
	r := func(region string) language.Region {
		return language.MustParseRegion(region)
	}
	t1800, _ := time.Parse("2006-01-02", "1800-01-01")
	type result struct {
		region   language.Region
		unit     Unit
		isTender bool
		from, to string
	}
	testCases := []struct {
		name    string
		opts    []QueryOption
		results []result
	}{{
		name:    "XA",
		opts:    []QueryOption{Region(r("XA"))},
		results: []result{},
	}, {
		name: "AC",
		opts: []QueryOption{Region(r("AC"))},
		results: []result{
			{r("AC"), MustParseISO("SHP"), true, "1976-01-01", ""},
		},
	}, {
		name: "US",
		opts: []QueryOption{Region(r("US"))},
		results: []result{
			{r("US"), MustParseISO("USD"), true, "1792-01-01", ""},
		},
	}, {
		name: "US-hist",
		opts: []QueryOption{Region(r("US")), Historical},
		results: []result{
			{r("US"), MustParseISO("USD"), true, "1792-01-01", ""},
		},
	}, {
		name: "US-non-tender",
		opts: []QueryOption{Region(r("US")), NonTender},
		results: []result{
			{r("US"), MustParseISO("USD"), true, "1792-01-01", ""},
			{r("US"), MustParseISO("USN"), false, "", ""},
		},
	}, {
		name: "US-historical+non-tender",
		opts: []QueryOption{Region(r("US")), Historical, NonTender},
		results: []result{
			{r("US"), MustParseISO("USD"), true, "1792-01-01", ""},
			{r("US"), MustParseISO("USN"), false, "", ""},
			{r("US"), MustParseISO("USS"), false, "", "2014-03-01"},
		},
	}, {
		name: "1800",
		opts: []QueryOption{Date(t1800)},
		results: []result{
			{r("CH"), MustParseISO("CHF"), true, "1799-03-17", ""},
			{r("GB"), MustParseISO("GBP"), true, "1694-07-27", ""},
			{r("GI"), MustParseISO("GIP"), true, "1713-01-01", ""},
			// The date for IE and PR seem wrong, so these may be updated at
			// some point causing the tests to fail.
			{r("IE"), MustParseISO("GBP"), true, "1800-01-01", "1922-01-01"},
			{r("PR"), MustParseISO("ESP"), true, "1800-01-01", "1898-12-10"},
			{r("US"), MustParseISO("USD"), true, "1792-01-01", ""},
		},
	}}
	for _, tc := range testCases {
		n := 0
		for it := Query(tc.opts...); it.Next(); n++ {
			if n < len(tc.results) {
				got := result{
					it.Region(),
					it.Unit(),
					it.IsTender(),
					getTime(it.From()),
					getTime(it.To()),
				}
				if got != tc.results[n] {
					t.Errorf("%s:%d: got %v; want %v", tc.name, n, got, tc.results[n])
				}
			}
		}
		if n != len(tc.results) {
			t.Errorf("%s: unexpected number of results: got %d; want %d", tc.name, n, len(tc.results))
		}
	}
}

func getTime(t time.Time, ok bool) string {
	if !ok {
		return ""
	}
	return t.Format("2006-01-02")
}
