// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package currency

import (
	"sort"
	"time"

	"golang.org/x/text/language"
)

// QueryIter represents a set of Units. The default set includes all Units that
// are currently in use as legal tender in any Region.
type QueryIter interface {
	// Next returns true if there is a next element available.
	// It must be called before any of the other methods are called.
	Next() bool

	// Unit returns the unit of the current iteration.
	Unit() Unit

	// Region returns the Region for the current iteration.
	Region() language.Region

	// From returns the date from which the unit was used in the region.
	// It returns false if this date is unknown.
	From() (time.Time, bool)

	// To returns the date up till which the unit was used in the region.
	// It returns false if this date is unknown or if the unit is still in use.
	To() (time.Time, bool)

	// IsTender reports whether the unit is a legal tender in the region during
	// the specified date range.
	IsTender() bool
}

// Query represents a set of Units. The default set includes all Units that are
// currently in use as legal tender in any Region.
func Query(options ...QueryOption) QueryIter {
	it := &iter{
		end:  len(regionData),
		date: 0xFFFFFFFF,
	}
	for _, fn := range options {
		fn(it)
	}
	return it
}

// NonTender returns a new query that also includes matching Units that are not
// legal tender.
var NonTender QueryOption = nonTender

func nonTender(i *iter) {
	i.nonTender = true
}

// Historical selects the units for all dates.
var Historical QueryOption = historical

func historical(i *iter) {
	i.date = hist
}

// A QueryOption can be used to change the set of unit information returned by
// a query.
type QueryOption func(*iter)

// Date queries the units that were in use at the given point in history.
func Date(t time.Time) QueryOption {
	d := toDate(t)
	return func(i *iter) {
		i.date = d
	}
}

// Region limits the query to only return entries for the given region.
func Region(r language.Region) QueryOption {
	p, end := len(regionData), len(regionData)
	x := regionToCode(r)
	i := sort.Search(len(regionData), func(i int) bool {
		return regionData[i].region >= x
	})
	if i < len(regionData) && regionData[i].region == x {
		p = i
		for i++; i < len(regionData) && regionData[i].region == x; i++ {
		}
		end = i
	}
	return func(i *iter) {
		i.p, i.end = p, end
	}
}

const (
	hist = 0x00
	now  = 0xFFFFFFFF
)

type iter struct {
	*regionInfo
	p, end    int
	date      uint32
	nonTender bool
}

func (i *iter) Next() bool {
	for ; i.p < i.end; i.p++ {
		i.regionInfo = &regionData[i.p]
		if !i.nonTender && !i.IsTender() {
			continue
		}
		if i.date == hist || (i.from <= i.date && (i.to == 0 || i.date <= i.to)) {
			i.p++
			return true
		}
	}
	return false
}

func (r *regionInfo) Region() language.Region {
	// TODO: this could be much faster.
	var buf [2]byte
	buf[0] = uint8(r.region >> 8)
	buf[1] = uint8(r.region)
	return language.MustParseRegion(string(buf[:]))
}

func (r *regionInfo) Unit() Unit {
	return Unit{r.code &^ nonTenderBit}
}

func (r *regionInfo) IsTender() bool {
	return r.code&nonTenderBit == 0
}

func (r *regionInfo) From() (time.Time, bool) {
	if r.from == 0 {
		return time.Time{}, false
	}
	return fromDate(r.from), true
}

func (r *regionInfo) To() (time.Time, bool) {
	if r.to == 0 {
		return time.Time{}, false
	}
	return fromDate(r.to), true
}
