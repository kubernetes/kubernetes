// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package cmp

import (
	"fmt"
	"reflect"

	"github.com/google/go-cmp/cmp/internal/value"
)

// TODO: Enforce limits?
//	* Enforce maximum number of records to print per node?
//	* Enforce maximum size in bytes allowed?
//	* As a heuristic, use less verbosity for equal nodes than unequal nodes.
// TODO: Enforce unique outputs?
//	* Avoid Stringer methods if it results in same output?
//	* Print pointer address if outputs still equal?

// numContextRecords is the number of surrounding equal records to print.
const numContextRecords = 2

type diffMode byte

const (
	diffUnknown   diffMode = 0
	diffIdentical diffMode = ' '
	diffRemoved   diffMode = '-'
	diffInserted  diffMode = '+'
)

type typeMode int

const (
	// emitType always prints the type.
	emitType typeMode = iota
	// elideType never prints the type.
	elideType
	// autoType prints the type only for composite kinds
	// (i.e., structs, slices, arrays, and maps).
	autoType
)

type formatOptions struct {
	// DiffMode controls the output mode of FormatDiff.
	//
	// If diffUnknown,   then produce a diff of the x and y values.
	// If diffIdentical, then emit values as if they were equal.
	// If diffRemoved,   then only emit x values (ignoring y values).
	// If diffInserted,  then only emit y values (ignoring x values).
	DiffMode diffMode

	// TypeMode controls whether to print the type for the current node.
	//
	// As a general rule of thumb, we always print the type of the next node
	// after an interface, and always elide the type of the next node after
	// a slice or map node.
	TypeMode typeMode

	// formatValueOptions are options specific to printing reflect.Values.
	formatValueOptions
}

func (opts formatOptions) WithDiffMode(d diffMode) formatOptions {
	opts.DiffMode = d
	return opts
}
func (opts formatOptions) WithTypeMode(t typeMode) formatOptions {
	opts.TypeMode = t
	return opts
}

// FormatDiff converts a valueNode tree into a textNode tree, where the later
// is a textual representation of the differences detected in the former.
func (opts formatOptions) FormatDiff(v *valueNode) textNode {
	// Check whether we have specialized formatting for this node.
	// This is not necessary, but helpful for producing more readable outputs.
	if opts.CanFormatDiffSlice(v) {
		return opts.FormatDiffSlice(v)
	}

	// For leaf nodes, format the value based on the reflect.Values alone.
	if v.MaxDepth == 0 {
		switch opts.DiffMode {
		case diffUnknown, diffIdentical:
			// Format Equal.
			if v.NumDiff == 0 {
				outx := opts.FormatValue(v.ValueX, visitedPointers{})
				outy := opts.FormatValue(v.ValueY, visitedPointers{})
				if v.NumIgnored > 0 && v.NumSame == 0 {
					return textEllipsis
				} else if outx.Len() < outy.Len() {
					return outx
				} else {
					return outy
				}
			}

			// Format unequal.
			assert(opts.DiffMode == diffUnknown)
			var list textList
			outx := opts.WithTypeMode(elideType).FormatValue(v.ValueX, visitedPointers{})
			outy := opts.WithTypeMode(elideType).FormatValue(v.ValueY, visitedPointers{})
			if outx != nil {
				list = append(list, textRecord{Diff: '-', Value: outx})
			}
			if outy != nil {
				list = append(list, textRecord{Diff: '+', Value: outy})
			}
			return opts.WithTypeMode(emitType).FormatType(v.Type, list)
		case diffRemoved:
			return opts.FormatValue(v.ValueX, visitedPointers{})
		case diffInserted:
			return opts.FormatValue(v.ValueY, visitedPointers{})
		default:
			panic("invalid diff mode")
		}
	}

	// Descend into the child value node.
	if v.TransformerName != "" {
		out := opts.WithTypeMode(emitType).FormatDiff(v.Value)
		out = textWrap{"Inverse(" + v.TransformerName + ", ", out, ")"}
		return opts.FormatType(v.Type, out)
	} else {
		switch k := v.Type.Kind(); k {
		case reflect.Struct, reflect.Array, reflect.Slice, reflect.Map:
			return opts.FormatType(v.Type, opts.formatDiffList(v.Records, k))
		case reflect.Ptr:
			return textWrap{"&", opts.FormatDiff(v.Value), ""}
		case reflect.Interface:
			return opts.WithTypeMode(emitType).FormatDiff(v.Value)
		default:
			panic(fmt.Sprintf("%v cannot have children", k))
		}
	}
}

func (opts formatOptions) formatDiffList(recs []reportRecord, k reflect.Kind) textNode {
	// Derive record name based on the data structure kind.
	var name string
	var formatKey func(reflect.Value) string
	switch k {
	case reflect.Struct:
		name = "field"
		opts = opts.WithTypeMode(autoType)
		formatKey = func(v reflect.Value) string { return v.String() }
	case reflect.Slice, reflect.Array:
		name = "element"
		opts = opts.WithTypeMode(elideType)
		formatKey = func(reflect.Value) string { return "" }
	case reflect.Map:
		name = "entry"
		opts = opts.WithTypeMode(elideType)
		formatKey = formatMapKey
	}

	// Handle unification.
	switch opts.DiffMode {
	case diffIdentical, diffRemoved, diffInserted:
		var list textList
		var deferredEllipsis bool // Add final "..." to indicate records were dropped
		for _, r := range recs {
			// Elide struct fields that are zero value.
			if k == reflect.Struct {
				var isZero bool
				switch opts.DiffMode {
				case diffIdentical:
					isZero = value.IsZero(r.Value.ValueX) || value.IsZero(r.Value.ValueX)
				case diffRemoved:
					isZero = value.IsZero(r.Value.ValueX)
				case diffInserted:
					isZero = value.IsZero(r.Value.ValueY)
				}
				if isZero {
					continue
				}
			}
			// Elide ignored nodes.
			if r.Value.NumIgnored > 0 && r.Value.NumSame+r.Value.NumDiff == 0 {
				deferredEllipsis = !(k == reflect.Slice || k == reflect.Array)
				if !deferredEllipsis {
					list.AppendEllipsis(diffStats{})
				}
				continue
			}
			if out := opts.FormatDiff(r.Value); out != nil {
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			}
		}
		if deferredEllipsis {
			list.AppendEllipsis(diffStats{})
		}
		return textWrap{"{", list, "}"}
	case diffUnknown:
	default:
		panic("invalid diff mode")
	}

	// Handle differencing.
	var list textList
	groups := coalesceAdjacentRecords(name, recs)
	for i, ds := range groups {
		// Handle equal records.
		if ds.NumDiff() == 0 {
			// Compute the number of leading and trailing records to print.
			var numLo, numHi int
			numEqual := ds.NumIgnored + ds.NumIdentical
			for numLo < numContextRecords && numLo+numHi < numEqual && i != 0 {
				if r := recs[numLo].Value; r.NumIgnored > 0 && r.NumSame+r.NumDiff == 0 {
					break
				}
				numLo++
			}
			for numHi < numContextRecords && numLo+numHi < numEqual && i != len(groups)-1 {
				if r := recs[numEqual-numHi-1].Value; r.NumIgnored > 0 && r.NumSame+r.NumDiff == 0 {
					break
				}
				numHi++
			}
			if numEqual-(numLo+numHi) == 1 && ds.NumIgnored == 0 {
				numHi++ // Avoid pointless coalescing of a single equal record
			}

			// Format the equal values.
			for _, r := range recs[:numLo] {
				out := opts.WithDiffMode(diffIdentical).FormatDiff(r.Value)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			}
			if numEqual > numLo+numHi {
				ds.NumIdentical -= numLo + numHi
				list.AppendEllipsis(ds)
			}
			for _, r := range recs[numEqual-numHi : numEqual] {
				out := opts.WithDiffMode(diffIdentical).FormatDiff(r.Value)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			}
			recs = recs[numEqual:]
			continue
		}

		// Handle unequal records.
		for _, r := range recs[:ds.NumDiff()] {
			switch {
			case opts.CanFormatDiffSlice(r.Value):
				out := opts.FormatDiffSlice(r.Value)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			case r.Value.NumChildren == r.Value.MaxDepth:
				outx := opts.WithDiffMode(diffRemoved).FormatDiff(r.Value)
				outy := opts.WithDiffMode(diffInserted).FormatDiff(r.Value)
				if outx != nil {
					list = append(list, textRecord{Diff: diffRemoved, Key: formatKey(r.Key), Value: outx})
				}
				if outy != nil {
					list = append(list, textRecord{Diff: diffInserted, Key: formatKey(r.Key), Value: outy})
				}
			default:
				out := opts.FormatDiff(r.Value)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			}
		}
		recs = recs[ds.NumDiff():]
	}
	assert(len(recs) == 0)
	return textWrap{"{", list, "}"}
}

// coalesceAdjacentRecords coalesces the list of records into groups of
// adjacent equal, or unequal counts.
func coalesceAdjacentRecords(name string, recs []reportRecord) (groups []diffStats) {
	var prevCase int // Arbitrary index into which case last occurred
	lastStats := func(i int) *diffStats {
		if prevCase != i {
			groups = append(groups, diffStats{Name: name})
			prevCase = i
		}
		return &groups[len(groups)-1]
	}
	for _, r := range recs {
		switch rv := r.Value; {
		case rv.NumIgnored > 0 && rv.NumSame+rv.NumDiff == 0:
			lastStats(1).NumIgnored++
		case rv.NumDiff == 0:
			lastStats(1).NumIdentical++
		case rv.NumDiff > 0 && !rv.ValueY.IsValid():
			lastStats(2).NumRemoved++
		case rv.NumDiff > 0 && !rv.ValueX.IsValid():
			lastStats(2).NumInserted++
		default:
			lastStats(2).NumModified++
		}
	}
	return groups
}
