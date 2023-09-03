// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp/internal/diff"
)

// CanFormatDiffSlice reports whether we support custom formatting for nodes
// that are slices of primitive kinds or strings.
func (opts formatOptions) CanFormatDiffSlice(v *valueNode) bool {
	switch {
	case opts.DiffMode != diffUnknown:
		return false // Must be formatting in diff mode
	case v.NumDiff == 0:
		return false // No differences detected
	case !v.ValueX.IsValid() || !v.ValueY.IsValid():
		return false // Both values must be valid
	case v.NumIgnored > 0:
		return false // Some ignore option was used
	case v.NumTransformed > 0:
		return false // Some transform option was used
	case v.NumCompared > 1:
		return false // More than one comparison was used
	case v.NumCompared == 1 && v.Type.Name() != "":
		// The need for cmp to check applicability of options on every element
		// in a slice is a significant performance detriment for large []byte.
		// The workaround is to specify Comparer(bytes.Equal),
		// which enables cmp to compare []byte more efficiently.
		// If they differ, we still want to provide batched diffing.
		// The logic disallows named types since they tend to have their own
		// String method, with nicer formatting than what this provides.
		return false
	}

	// Check whether this is an interface with the same concrete types.
	t := v.Type
	vx, vy := v.ValueX, v.ValueY
	if t.Kind() == reflect.Interface && !vx.IsNil() && !vy.IsNil() && vx.Elem().Type() == vy.Elem().Type() {
		vx, vy = vx.Elem(), vy.Elem()
		t = vx.Type()
	}

	// Check whether we provide specialized diffing for this type.
	switch t.Kind() {
	case reflect.String:
	case reflect.Array, reflect.Slice:
		// Only slices of primitive types have specialized handling.
		switch t.Elem().Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
			reflect.Bool, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		default:
			return false
		}

		// Both slice values have to be non-empty.
		if t.Kind() == reflect.Slice && (vx.Len() == 0 || vy.Len() == 0) {
			return false
		}

		// If a sufficient number of elements already differ,
		// use specialized formatting even if length requirement is not met.
		if v.NumDiff > v.NumSame {
			return true
		}
	default:
		return false
	}

	// Use specialized string diffing for longer slices or strings.
	const minLength = 32
	return vx.Len() >= minLength && vy.Len() >= minLength
}

// FormatDiffSlice prints a diff for the slices (or strings) represented by v.
// This provides custom-tailored logic to make printing of differences in
// textual strings and slices of primitive kinds more readable.
func (opts formatOptions) FormatDiffSlice(v *valueNode) textNode {
	assert(opts.DiffMode == diffUnknown)
	t, vx, vy := v.Type, v.ValueX, v.ValueY
	if t.Kind() == reflect.Interface {
		vx, vy = vx.Elem(), vy.Elem()
		t = vx.Type()
		opts = opts.WithTypeMode(emitType)
	}

	// Auto-detect the type of the data.
	var sx, sy string
	var ssx, ssy []string
	var isString, isMostlyText, isPureLinedText, isBinary bool
	switch {
	case t.Kind() == reflect.String:
		sx, sy = vx.String(), vy.String()
		isString = true
	case t.Kind() == reflect.Slice && t.Elem() == byteType:
		sx, sy = string(vx.Bytes()), string(vy.Bytes())
		isString = true
	case t.Kind() == reflect.Array:
		// Arrays need to be addressable for slice operations to work.
		vx2, vy2 := reflect.New(t).Elem(), reflect.New(t).Elem()
		vx2.Set(vx)
		vy2.Set(vy)
		vx, vy = vx2, vy2
	}
	if isString {
		var numTotalRunes, numValidRunes, numLines, lastLineIdx, maxLineLen int
		for i, r := range sx + sy {
			numTotalRunes++
			if (unicode.IsPrint(r) || unicode.IsSpace(r)) && r != utf8.RuneError {
				numValidRunes++
			}
			if r == '\n' {
				if maxLineLen < i-lastLineIdx {
					maxLineLen = i - lastLineIdx
				}
				lastLineIdx = i + 1
				numLines++
			}
		}
		isPureText := numValidRunes == numTotalRunes
		isMostlyText = float64(numValidRunes) > math.Floor(0.90*float64(numTotalRunes))
		isPureLinedText = isPureText && numLines >= 4 && maxLineLen <= 1024
		isBinary = !isMostlyText

		// Avoid diffing by lines if it produces a significantly more complex
		// edit script than diffing by bytes.
		if isPureLinedText {
			ssx = strings.Split(sx, "\n")
			ssy = strings.Split(sy, "\n")
			esLines := diff.Difference(len(ssx), len(ssy), func(ix, iy int) diff.Result {
				return diff.BoolResult(ssx[ix] == ssy[iy])
			})
			esBytes := diff.Difference(len(sx), len(sy), func(ix, iy int) diff.Result {
				return diff.BoolResult(sx[ix] == sy[iy])
			})
			efficiencyLines := float64(esLines.Dist()) / float64(len(esLines))
			efficiencyBytes := float64(esBytes.Dist()) / float64(len(esBytes))
			quotedLength := len(strconv.Quote(sx + sy))
			unquotedLength := len(sx) + len(sy)
			escapeExpansionRatio := float64(quotedLength) / float64(unquotedLength)
			isPureLinedText = efficiencyLines < 4*efficiencyBytes || escapeExpansionRatio > 1.1
		}
	}

	// Format the string into printable records.
	var list textList
	var delim string
	switch {
	// If the text appears to be multi-lined text,
	// then perform differencing across individual lines.
	case isPureLinedText:
		list = opts.formatDiffSlice(
			reflect.ValueOf(ssx), reflect.ValueOf(ssy), 1, "line",
			func(v reflect.Value, d diffMode) textRecord {
				s := formatString(v.Index(0).String())
				return textRecord{Diff: d, Value: textLine(s)}
			},
		)
		delim = "\n"

		// If possible, use a custom triple-quote (""") syntax for printing
		// differences in a string literal. This format is more readable,
		// but has edge-cases where differences are visually indistinguishable.
		// This format is avoided under the following conditions:
		//   - A line starts with `"""`
		//   - A line starts with "..."
		//   - A line contains non-printable characters
		//   - Adjacent different lines differ only by whitespace
		//
		// For example:
		//
		//		"""
		//		... // 3 identical lines
		//		foo
		//		bar
		//	-	baz
		//	+	BAZ
		//		"""
		isTripleQuoted := true
		prevRemoveLines := map[string]bool{}
		prevInsertLines := map[string]bool{}
		var list2 textList
		list2 = append(list2, textRecord{Value: textLine(`"""`), ElideComma: true})
		for _, r := range list {
			if !r.Value.Equal(textEllipsis) {
				line, _ := strconv.Unquote(string(r.Value.(textLine)))
				line = strings.TrimPrefix(strings.TrimSuffix(line, "\r"), "\r") // trim leading/trailing carriage returns for legacy Windows endline support
				normLine := strings.Map(func(r rune) rune {
					if unicode.IsSpace(r) {
						return -1 // drop whitespace to avoid visually indistinguishable output
					}
					return r
				}, line)
				isPrintable := func(r rune) bool {
					return unicode.IsPrint(r) || r == '\t' // specially treat tab as printable
				}
				isTripleQuoted = !strings.HasPrefix(line, `"""`) && !strings.HasPrefix(line, "...") && strings.TrimFunc(line, isPrintable) == ""
				switch r.Diff {
				case diffRemoved:
					isTripleQuoted = isTripleQuoted && !prevInsertLines[normLine]
					prevRemoveLines[normLine] = true
				case diffInserted:
					isTripleQuoted = isTripleQuoted && !prevRemoveLines[normLine]
					prevInsertLines[normLine] = true
				}
				if !isTripleQuoted {
					break
				}
				r.Value = textLine(line)
				r.ElideComma = true
			}
			if !(r.Diff == diffRemoved || r.Diff == diffInserted) { // start a new non-adjacent difference group
				prevRemoveLines = map[string]bool{}
				prevInsertLines = map[string]bool{}
			}
			list2 = append(list2, r)
		}
		if r := list2[len(list2)-1]; r.Diff == diffIdentical && len(r.Value.(textLine)) == 0 {
			list2 = list2[:len(list2)-1] // elide single empty line at the end
		}
		list2 = append(list2, textRecord{Value: textLine(`"""`), ElideComma: true})
		if isTripleQuoted {
			var out textNode = &textWrap{Prefix: "(", Value: list2, Suffix: ")"}
			switch t.Kind() {
			case reflect.String:
				if t != stringType {
					out = opts.FormatType(t, out)
				}
			case reflect.Slice:
				// Always emit type for slices since the triple-quote syntax
				// looks like a string (not a slice).
				opts = opts.WithTypeMode(emitType)
				out = opts.FormatType(t, out)
			}
			return out
		}

	// If the text appears to be single-lined text,
	// then perform differencing in approximately fixed-sized chunks.
	// The output is printed as quoted strings.
	case isMostlyText:
		list = opts.formatDiffSlice(
			reflect.ValueOf(sx), reflect.ValueOf(sy), 64, "byte",
			func(v reflect.Value, d diffMode) textRecord {
				s := formatString(v.String())
				return textRecord{Diff: d, Value: textLine(s)}
			},
		)

	// If the text appears to be binary data,
	// then perform differencing in approximately fixed-sized chunks.
	// The output is inspired by hexdump.
	case isBinary:
		list = opts.formatDiffSlice(
			reflect.ValueOf(sx), reflect.ValueOf(sy), 16, "byte",
			func(v reflect.Value, d diffMode) textRecord {
				var ss []string
				for i := 0; i < v.Len(); i++ {
					ss = append(ss, formatHex(v.Index(i).Uint()))
				}
				s := strings.Join(ss, ", ")
				comment := commentString(fmt.Sprintf("%c|%v|", d, formatASCII(v.String())))
				return textRecord{Diff: d, Value: textLine(s), Comment: comment}
			},
		)

	// For all other slices of primitive types,
	// then perform differencing in approximately fixed-sized chunks.
	// The size of each chunk depends on the width of the element kind.
	default:
		var chunkSize int
		if t.Elem().Kind() == reflect.Bool {
			chunkSize = 16
		} else {
			switch t.Elem().Bits() {
			case 8:
				chunkSize = 16
			case 16:
				chunkSize = 12
			case 32:
				chunkSize = 8
			default:
				chunkSize = 8
			}
		}
		list = opts.formatDiffSlice(
			vx, vy, chunkSize, t.Elem().Kind().String(),
			func(v reflect.Value, d diffMode) textRecord {
				var ss []string
				for i := 0; i < v.Len(); i++ {
					switch t.Elem().Kind() {
					case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
						ss = append(ss, fmt.Sprint(v.Index(i).Int()))
					case reflect.Uint, reflect.Uint16, reflect.Uint32, reflect.Uint64:
						ss = append(ss, fmt.Sprint(v.Index(i).Uint()))
					case reflect.Uint8, reflect.Uintptr:
						ss = append(ss, formatHex(v.Index(i).Uint()))
					case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
						ss = append(ss, fmt.Sprint(v.Index(i).Interface()))
					}
				}
				s := strings.Join(ss, ", ")
				return textRecord{Diff: d, Value: textLine(s)}
			},
		)
	}

	// Wrap the output with appropriate type information.
	var out textNode = &textWrap{Prefix: "{", Value: list, Suffix: "}"}
	if !isMostlyText {
		// The "{...}" byte-sequence literal is not valid Go syntax for strings.
		// Emit the type for extra clarity (e.g. "string{...}").
		if t.Kind() == reflect.String {
			opts = opts.WithTypeMode(emitType)
		}
		return opts.FormatType(t, out)
	}
	switch t.Kind() {
	case reflect.String:
		out = &textWrap{Prefix: "strings.Join(", Value: out, Suffix: fmt.Sprintf(", %q)", delim)}
		if t != stringType {
			out = opts.FormatType(t, out)
		}
	case reflect.Slice:
		out = &textWrap{Prefix: "bytes.Join(", Value: out, Suffix: fmt.Sprintf(", %q)", delim)}
		if t != bytesType {
			out = opts.FormatType(t, out)
		}
	}
	return out
}

// formatASCII formats s as an ASCII string.
// This is useful for printing binary strings in a semi-legible way.
func formatASCII(s string) string {
	b := bytes.Repeat([]byte{'.'}, len(s))
	for i := 0; i < len(s); i++ {
		if ' ' <= s[i] && s[i] <= '~' {
			b[i] = s[i]
		}
	}
	return string(b)
}

func (opts formatOptions) formatDiffSlice(
	vx, vy reflect.Value, chunkSize int, name string,
	makeRec func(reflect.Value, diffMode) textRecord,
) (list textList) {
	eq := func(ix, iy int) bool {
		return vx.Index(ix).Interface() == vy.Index(iy).Interface()
	}
	es := diff.Difference(vx.Len(), vy.Len(), func(ix, iy int) diff.Result {
		return diff.BoolResult(eq(ix, iy))
	})

	appendChunks := func(v reflect.Value, d diffMode) int {
		n0 := v.Len()
		for v.Len() > 0 {
			n := chunkSize
			if n > v.Len() {
				n = v.Len()
			}
			list = append(list, makeRec(v.Slice(0, n), d))
			v = v.Slice(n, v.Len())
		}
		return n0 - v.Len()
	}

	var numDiffs int
	maxLen := -1
	if opts.LimitVerbosity {
		maxLen = (1 << opts.verbosity()) << 2 // 4, 8, 16, 32, 64, etc...
		opts.VerbosityLevel--
	}

	groups := coalesceAdjacentEdits(name, es)
	groups = coalesceInterveningIdentical(groups, chunkSize/4)
	groups = cleanupSurroundingIdentical(groups, eq)
	maxGroup := diffStats{Name: name}
	for i, ds := range groups {
		if maxLen >= 0 && numDiffs >= maxLen {
			maxGroup = maxGroup.Append(ds)
			continue
		}

		// Print equal.
		if ds.NumDiff() == 0 {
			// Compute the number of leading and trailing equal bytes to print.
			var numLo, numHi int
			numEqual := ds.NumIgnored + ds.NumIdentical
			for numLo < chunkSize*numContextRecords && numLo+numHi < numEqual && i != 0 {
				numLo++
			}
			for numHi < chunkSize*numContextRecords && numLo+numHi < numEqual && i != len(groups)-1 {
				numHi++
			}
			if numEqual-(numLo+numHi) <= chunkSize && ds.NumIgnored == 0 {
				numHi = numEqual - numLo // Avoid pointless coalescing of single equal row
			}

			// Print the equal bytes.
			appendChunks(vx.Slice(0, numLo), diffIdentical)
			if numEqual > numLo+numHi {
				ds.NumIdentical -= numLo + numHi
				list.AppendEllipsis(ds)
			}
			appendChunks(vx.Slice(numEqual-numHi, numEqual), diffIdentical)
			vx = vx.Slice(numEqual, vx.Len())
			vy = vy.Slice(numEqual, vy.Len())
			continue
		}

		// Print unequal.
		len0 := len(list)
		nx := appendChunks(vx.Slice(0, ds.NumIdentical+ds.NumRemoved+ds.NumModified), diffRemoved)
		vx = vx.Slice(nx, vx.Len())
		ny := appendChunks(vy.Slice(0, ds.NumIdentical+ds.NumInserted+ds.NumModified), diffInserted)
		vy = vy.Slice(ny, vy.Len())
		numDiffs += len(list) - len0
	}
	if maxGroup.IsZero() {
		assert(vx.Len() == 0 && vy.Len() == 0)
	} else {
		list.AppendEllipsis(maxGroup)
	}
	return list
}

// coalesceAdjacentEdits coalesces the list of edits into groups of adjacent
// equal or unequal counts.
//
// Example:
//
//	Input:  "..XXY...Y"
//	Output: [
//		{NumIdentical: 2},
//		{NumRemoved: 2, NumInserted 1},
//		{NumIdentical: 3},
//		{NumInserted: 1},
//	]
func coalesceAdjacentEdits(name string, es diff.EditScript) (groups []diffStats) {
	var prevMode byte
	lastStats := func(mode byte) *diffStats {
		if prevMode != mode {
			groups = append(groups, diffStats{Name: name})
			prevMode = mode
		}
		return &groups[len(groups)-1]
	}
	for _, e := range es {
		switch e {
		case diff.Identity:
			lastStats('=').NumIdentical++
		case diff.UniqueX:
			lastStats('!').NumRemoved++
		case diff.UniqueY:
			lastStats('!').NumInserted++
		case diff.Modified:
			lastStats('!').NumModified++
		}
	}
	return groups
}

// coalesceInterveningIdentical coalesces sufficiently short (<= windowSize)
// equal groups into adjacent unequal groups that currently result in a
// dual inserted/removed printout. This acts as a high-pass filter to smooth
// out high-frequency changes within the windowSize.
//
// Example:
//
//	WindowSize: 16,
//	Input: [
//		{NumIdentical: 61},              // group 0
//		{NumRemoved: 3, NumInserted: 1}, // group 1
//		{NumIdentical: 6},               // ├── coalesce
//		{NumInserted: 2},                // ├── coalesce
//		{NumIdentical: 1},               // ├── coalesce
//		{NumRemoved: 9},                 // └── coalesce
//		{NumIdentical: 64},              // group 2
//		{NumRemoved: 3, NumInserted: 1}, // group 3
//		{NumIdentical: 6},               // ├── coalesce
//		{NumInserted: 2},                // ├── coalesce
//		{NumIdentical: 1},               // ├── coalesce
//		{NumRemoved: 7},                 // ├── coalesce
//		{NumIdentical: 1},               // ├── coalesce
//		{NumRemoved: 2},                 // └── coalesce
//		{NumIdentical: 63},              // group 4
//	]
//	Output: [
//		{NumIdentical: 61},
//		{NumIdentical: 7, NumRemoved: 12, NumInserted: 3},
//		{NumIdentical: 64},
//		{NumIdentical: 8, NumRemoved: 12, NumInserted: 3},
//		{NumIdentical: 63},
//	]
func coalesceInterveningIdentical(groups []diffStats, windowSize int) []diffStats {
	groups, groupsOrig := groups[:0], groups
	for i, ds := range groupsOrig {
		if len(groups) >= 2 && ds.NumDiff() > 0 {
			prev := &groups[len(groups)-2] // Unequal group
			curr := &groups[len(groups)-1] // Equal group
			next := &groupsOrig[i]         // Unequal group
			hadX, hadY := prev.NumRemoved > 0, prev.NumInserted > 0
			hasX, hasY := next.NumRemoved > 0, next.NumInserted > 0
			if ((hadX || hasX) && (hadY || hasY)) && curr.NumIdentical <= windowSize {
				*prev = prev.Append(*curr).Append(*next)
				groups = groups[:len(groups)-1] // Truncate off equal group
				continue
			}
		}
		groups = append(groups, ds)
	}
	return groups
}

// cleanupSurroundingIdentical scans through all unequal groups, and
// moves any leading sequence of equal elements to the preceding equal group and
// moves and trailing sequence of equal elements to the succeeding equal group.
//
// This is necessary since coalesceInterveningIdentical may coalesce edit groups
// together such that leading/trailing spans of equal elements becomes possible.
// Note that this can occur even with an optimal diffing algorithm.
//
// Example:
//
//	Input: [
//		{NumIdentical: 61},
//		{NumIdentical: 1 , NumRemoved: 11, NumInserted: 2}, // assume 3 leading identical elements
//		{NumIdentical: 67},
//		{NumIdentical: 7, NumRemoved: 12, NumInserted: 3},  // assume 10 trailing identical elements
//		{NumIdentical: 54},
//	]
//	Output: [
//		{NumIdentical: 64}, // incremented by 3
//		{NumRemoved: 9},
//		{NumIdentical: 67},
//		{NumRemoved: 9},
//		{NumIdentical: 64}, // incremented by 10
//	]
func cleanupSurroundingIdentical(groups []diffStats, eq func(i, j int) bool) []diffStats {
	var ix, iy int // indexes into sequence x and y
	for i, ds := range groups {
		// Handle equal group.
		if ds.NumDiff() == 0 {
			ix += ds.NumIdentical
			iy += ds.NumIdentical
			continue
		}

		// Handle unequal group.
		nx := ds.NumIdentical + ds.NumRemoved + ds.NumModified
		ny := ds.NumIdentical + ds.NumInserted + ds.NumModified
		var numLeadingIdentical, numTrailingIdentical int
		for j := 0; j < nx && j < ny && eq(ix+j, iy+j); j++ {
			numLeadingIdentical++
		}
		for j := 0; j < nx && j < ny && eq(ix+nx-1-j, iy+ny-1-j); j++ {
			numTrailingIdentical++
		}
		if numIdentical := numLeadingIdentical + numTrailingIdentical; numIdentical > 0 {
			if numLeadingIdentical > 0 {
				// Remove leading identical span from this group and
				// insert it into the preceding group.
				if i-1 >= 0 {
					groups[i-1].NumIdentical += numLeadingIdentical
				} else {
					// No preceding group exists, so prepend a new group,
					// but do so after we finish iterating over all groups.
					defer func() {
						groups = append([]diffStats{{Name: groups[0].Name, NumIdentical: numLeadingIdentical}}, groups...)
					}()
				}
				// Increment indexes since the preceding group would have handled this.
				ix += numLeadingIdentical
				iy += numLeadingIdentical
			}
			if numTrailingIdentical > 0 {
				// Remove trailing identical span from this group and
				// insert it into the succeeding group.
				if i+1 < len(groups) {
					groups[i+1].NumIdentical += numTrailingIdentical
				} else {
					// No succeeding group exists, so append a new group,
					// but do so after we finish iterating over all groups.
					defer func() {
						groups = append(groups, diffStats{Name: groups[len(groups)-1].Name, NumIdentical: numTrailingIdentical})
					}()
				}
				// Do not increment indexes since the succeeding group will handle this.
			}

			// Update this group since some identical elements were removed.
			nx -= numIdentical
			ny -= numIdentical
			groups[i] = diffStats{Name: ds.Name, NumRemoved: nx, NumInserted: ny}
		}
		ix += nx
		iy += ny
	}
	return groups
}
