// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import (
	"fmt"
	"reflect"

	"github.com/google/go-cmp/cmp/internal/value"
)

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
func (opts formatOptions) WithVerbosity(level int) formatOptions {
	opts.VerbosityLevel = level
	opts.LimitVerbosity = true
	return opts
}
func (opts formatOptions) verbosity() uint {
	switch {
	case opts.VerbosityLevel < 0:
		return 0
	case opts.VerbosityLevel > 16:
		return 16 // some reasonable maximum to avoid shift overflow
	default:
		return uint(opts.VerbosityLevel)
	}
}

const maxVerbosityPreset = 6

// verbosityPreset modifies the verbosity settings given an index
// between 0 and maxVerbosityPreset, inclusive.
func verbosityPreset(opts formatOptions, i int) formatOptions {
	opts.VerbosityLevel = int(opts.verbosity()) + 2*i
	if i > 0 {
		opts.AvoidStringer = true
	}
	if i >= maxVerbosityPreset {
		opts.PrintAddresses = true
		opts.QualifiedNames = true
	}
	return opts
}

// FormatDiff converts a valueNode tree into a textNode tree, where the later
// is a textual representation of the differences detected in the former.
func (opts formatOptions) FormatDiff(v *valueNode, ptrs *pointerReferences) (out textNode) {
	if opts.DiffMode == diffIdentical {
		opts = opts.WithVerbosity(1)
	} else if opts.verbosity() < 3 {
		opts = opts.WithVerbosity(3)
	}

	// Check whether we have specialized formatting for this node.
	// This is not necessary, but helpful for producing more readable outputs.
	if opts.CanFormatDiffSlice(v) {
		return opts.FormatDiffSlice(v)
	}

	var parentKind reflect.Kind
	if v.parent != nil && v.parent.TransformerName == "" {
		parentKind = v.parent.Type.Kind()
	}

	// For leaf nodes, format the value based on the reflect.Values alone.
	if v.MaxDepth == 0 {
		switch opts.DiffMode {
		case diffUnknown, diffIdentical:
			// Format Equal.
			if v.NumDiff == 0 {
				outx := opts.FormatValue(v.ValueX, parentKind, ptrs)
				outy := opts.FormatValue(v.ValueY, parentKind, ptrs)
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
			outx := opts.WithTypeMode(elideType).FormatValue(v.ValueX, parentKind, ptrs)
			outy := opts.WithTypeMode(elideType).FormatValue(v.ValueY, parentKind, ptrs)
			for i := 0; i <= maxVerbosityPreset && outx != nil && outy != nil && outx.Equal(outy); i++ {
				opts2 := verbosityPreset(opts, i).WithTypeMode(elideType)
				outx = opts2.FormatValue(v.ValueX, parentKind, ptrs)
				outy = opts2.FormatValue(v.ValueY, parentKind, ptrs)
			}
			if outx != nil {
				list = append(list, textRecord{Diff: '-', Value: outx})
			}
			if outy != nil {
				list = append(list, textRecord{Diff: '+', Value: outy})
			}
			return opts.WithTypeMode(emitType).FormatType(v.Type, list)
		case diffRemoved:
			return opts.FormatValue(v.ValueX, parentKind, ptrs)
		case diffInserted:
			return opts.FormatValue(v.ValueY, parentKind, ptrs)
		default:
			panic("invalid diff mode")
		}
	}

	// Register slice element to support cycle detection.
	if parentKind == reflect.Slice {
		ptrRefs := ptrs.PushPair(v.ValueX, v.ValueY, opts.DiffMode, true)
		defer ptrs.Pop()
		defer func() { out = wrapTrunkReferences(ptrRefs, out) }()
	}

	// Descend into the child value node.
	if v.TransformerName != "" {
		out := opts.WithTypeMode(emitType).FormatDiff(v.Value, ptrs)
		out = &textWrap{Prefix: "Inverse(" + v.TransformerName + ", ", Value: out, Suffix: ")"}
		return opts.FormatType(v.Type, out)
	} else {
		switch k := v.Type.Kind(); k {
		case reflect.Struct, reflect.Array, reflect.Slice:
			out = opts.formatDiffList(v.Records, k, ptrs)
			out = opts.FormatType(v.Type, out)
		case reflect.Map:
			// Register map to support cycle detection.
			ptrRefs := ptrs.PushPair(v.ValueX, v.ValueY, opts.DiffMode, false)
			defer ptrs.Pop()

			out = opts.formatDiffList(v.Records, k, ptrs)
			out = wrapTrunkReferences(ptrRefs, out)
			out = opts.FormatType(v.Type, out)
		case reflect.Ptr:
			// Register pointer to support cycle detection.
			ptrRefs := ptrs.PushPair(v.ValueX, v.ValueY, opts.DiffMode, false)
			defer ptrs.Pop()

			out = opts.FormatDiff(v.Value, ptrs)
			out = wrapTrunkReferences(ptrRefs, out)
			out = &textWrap{Prefix: "&", Value: out}
		case reflect.Interface:
			out = opts.WithTypeMode(emitType).FormatDiff(v.Value, ptrs)
		default:
			panic(fmt.Sprintf("%v cannot have children", k))
		}
		return out
	}
}

func (opts formatOptions) formatDiffList(recs []reportRecord, k reflect.Kind, ptrs *pointerReferences) textNode {
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
		formatKey = func(v reflect.Value) string { return formatMapKey(v, false, ptrs) }
	}

	maxLen := -1
	if opts.LimitVerbosity {
		if opts.DiffMode == diffIdentical {
			maxLen = ((1 << opts.verbosity()) >> 1) << 2 // 0, 4, 8, 16, 32, etc...
		} else {
			maxLen = (1 << opts.verbosity()) << 1 // 2, 4, 8, 16, 32, 64, etc...
		}
		opts.VerbosityLevel--
	}

	// Handle unification.
	switch opts.DiffMode {
	case diffIdentical, diffRemoved, diffInserted:
		var list textList
		var deferredEllipsis bool // Add final "..." to indicate records were dropped
		for _, r := range recs {
			if len(list) == maxLen {
				deferredEllipsis = true
				break
			}

			// Elide struct fields that are zero value.
			if k == reflect.Struct {
				var isZero bool
				switch opts.DiffMode {
				case diffIdentical:
					isZero = value.IsZero(r.Value.ValueX) || value.IsZero(r.Value.ValueY)
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
			if out := opts.FormatDiff(r.Value, ptrs); out != nil {
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
			}
		}
		if deferredEllipsis {
			list.AppendEllipsis(diffStats{})
		}
		return &textWrap{Prefix: "{", Value: list, Suffix: "}"}
	case diffUnknown:
	default:
		panic("invalid diff mode")
	}

	// Handle differencing.
	var numDiffs int
	var list textList
	var keys []reflect.Value // invariant: len(list) == len(keys)
	groups := coalesceAdjacentRecords(name, recs)
	maxGroup := diffStats{Name: name}
	for i, ds := range groups {
		if maxLen >= 0 && numDiffs >= maxLen {
			maxGroup = maxGroup.Append(ds)
			continue
		}

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
				out := opts.WithDiffMode(diffIdentical).FormatDiff(r.Value, ptrs)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
				keys = append(keys, r.Key)
			}
			if numEqual > numLo+numHi {
				ds.NumIdentical -= numLo + numHi
				list.AppendEllipsis(ds)
				for len(keys) < len(list) {
					keys = append(keys, reflect.Value{})
				}
			}
			for _, r := range recs[numEqual-numHi : numEqual] {
				out := opts.WithDiffMode(diffIdentical).FormatDiff(r.Value, ptrs)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
				keys = append(keys, r.Key)
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
				keys = append(keys, r.Key)
			case r.Value.NumChildren == r.Value.MaxDepth:
				outx := opts.WithDiffMode(diffRemoved).FormatDiff(r.Value, ptrs)
				outy := opts.WithDiffMode(diffInserted).FormatDiff(r.Value, ptrs)
				for i := 0; i <= maxVerbosityPreset && outx != nil && outy != nil && outx.Equal(outy); i++ {
					opts2 := verbosityPreset(opts, i)
					outx = opts2.WithDiffMode(diffRemoved).FormatDiff(r.Value, ptrs)
					outy = opts2.WithDiffMode(diffInserted).FormatDiff(r.Value, ptrs)
				}
				if outx != nil {
					list = append(list, textRecord{Diff: diffRemoved, Key: formatKey(r.Key), Value: outx})
					keys = append(keys, r.Key)
				}
				if outy != nil {
					list = append(list, textRecord{Diff: diffInserted, Key: formatKey(r.Key), Value: outy})
					keys = append(keys, r.Key)
				}
			default:
				out := opts.FormatDiff(r.Value, ptrs)
				list = append(list, textRecord{Key: formatKey(r.Key), Value: out})
				keys = append(keys, r.Key)
			}
		}
		recs = recs[ds.NumDiff():]
		numDiffs += ds.NumDiff()
	}
	if maxGroup.IsZero() {
		assert(len(recs) == 0)
	} else {
		list.AppendEllipsis(maxGroup)
		for len(keys) < len(list) {
			keys = append(keys, reflect.Value{})
		}
	}
	assert(len(list) == len(keys))

	// For maps, the default formatting logic uses fmt.Stringer which may
	// produce ambiguous output. Avoid calling String to disambiguate.
	if k == reflect.Map {
		var ambiguous bool
		seenKeys := map[string]reflect.Value{}
		for i, currKey := range keys {
			if currKey.IsValid() {
				strKey := list[i].Key
				prevKey, seen := seenKeys[strKey]
				if seen && prevKey.CanInterface() && currKey.CanInterface() {
					ambiguous = prevKey.Interface() != currKey.Interface()
					if ambiguous {
						break
					}
				}
				seenKeys[strKey] = currKey
			}
		}
		if ambiguous {
			for i, k := range keys {
				if k.IsValid() {
					list[i].Key = formatMapKey(k, true, ptrs)
				}
			}
		}
	}

	return &textWrap{Prefix: "{", Value: list, Suffix: "}"}
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
