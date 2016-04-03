/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bigtable

import (
	"fmt"
	"strings"

	btdpb "google.golang.org/cloud/bigtable/internal/data_proto"
)

// A Filter represents a row filter.
type Filter interface {
	String() string
	proto() *btdpb.RowFilter
}

// ChainFilters returns a filter that applies a sequence of filters.
func ChainFilters(sub ...Filter) Filter { return chainFilter{sub} }

type chainFilter struct {
	sub []Filter
}

func (cf chainFilter) String() string {
	var ss []string
	for _, sf := range cf.sub {
		ss = append(ss, sf.String())
	}
	return "(" + strings.Join(ss, " | ") + ")"
}

func (cf chainFilter) proto() *btdpb.RowFilter {
	chain := &btdpb.RowFilter_Chain{}
	for _, sf := range cf.sub {
		chain.Filters = append(chain.Filters, sf.proto())
	}
	return &btdpb.RowFilter{
		Filter: &btdpb.RowFilter_Chain_{chain},
	}
}

// InterleaveFilters returns a filter that applies a set of filters in parallel
// and interleaves the results.
func InterleaveFilters(sub ...Filter) Filter { return interleaveFilter{sub} }

type interleaveFilter struct {
	sub []Filter
}

func (ilf interleaveFilter) String() string {
	var ss []string
	for _, sf := range ilf.sub {
		ss = append(ss, sf.String())
	}
	return "(" + strings.Join(ss, " + ") + ")"
}

func (ilf interleaveFilter) proto() *btdpb.RowFilter {
	inter := &btdpb.RowFilter_Interleave{}
	for _, sf := range ilf.sub {
		inter.Filters = append(inter.Filters, sf.proto())
	}
	return &btdpb.RowFilter{
		Filter: &btdpb.RowFilter_Interleave_{inter},
	}
}

// RowKeyFilter returns a filter that matches cells from rows whose
// key matches the provided RE2 pattern.
// See https://github.com/google/re2/wiki/Syntax for the accepted syntax.
func RowKeyFilter(pattern string) Filter { return rowKeyFilter(pattern) }

type rowKeyFilter string

func (rkf rowKeyFilter) String() string { return fmt.Sprintf("row(%s)", string(rkf)) }

func (rkf rowKeyFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_RowKeyRegexFilter{[]byte(rkf)}}
}

// FamilyFilter returns a filter that matches cells whose family name
// matches the provided RE2 pattern.
// See https://github.com/google/re2/wiki/Syntax for the accepted syntax.
func FamilyFilter(pattern string) Filter { return familyFilter(pattern) }

type familyFilter string

func (ff familyFilter) String() string { return fmt.Sprintf("col(%s:)", string(ff)) }

func (ff familyFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_FamilyNameRegexFilter{string(ff)}}
}

// ColumnFilter returns a filter that matches cells whose column name
// matches the provided RE2 pattern.
// See https://github.com/google/re2/wiki/Syntax for the accepted syntax.
func ColumnFilter(pattern string) Filter { return columnFilter(pattern) }

type columnFilter string

func (cf columnFilter) String() string { return fmt.Sprintf("col(.*:%s)", string(cf)) }

func (cf columnFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_ColumnQualifierRegexFilter{[]byte(cf)}}
}

// ValueFilter returns a filter that matches cells whose value
// matches the provided RE2 pattern.
// See https://github.com/google/re2/wiki/Syntax for the accepted syntax.
func ValueFilter(pattern string) Filter { return valueFilter(pattern) }

type valueFilter string

func (vf valueFilter) String() string { return fmt.Sprintf("value_match(%s)", string(vf)) }

func (vf valueFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_ValueRegexFilter{[]byte(vf)}}
}

// LatestNFilter returns a filter that matches the most recent N cells in each column.
func LatestNFilter(n int) Filter { return latestNFilter(n) }

type latestNFilter int32

func (lnf latestNFilter) String() string { return fmt.Sprintf("col(*,%d)", lnf) }

func (lnf latestNFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_CellsPerColumnLimitFilter{int32(lnf)}}
}

// StripValueFilter returns a filter that replaces each value with the empty string.
func StripValueFilter() Filter { return stripValueFilter{} }

type stripValueFilter struct{}

func (stripValueFilter) String() string { return "strip_value()" }
func (stripValueFilter) proto() *btdpb.RowFilter {
	return &btdpb.RowFilter{Filter: &btdpb.RowFilter_StripValueTransformer{true}}
}

// TODO(dsymonds): More filters: cond, col/ts/value range, sampling
