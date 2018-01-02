// Copyright (c) 2012-2016 The go-diff authors. All rights reserved.
// https://github.com/sergi/go-diff
// See the included LICENSE file for license details.
//
// go-diff is a Go implementation of Google's Diff, Match, and Patch library
// Original library is Copyright (c) 2006 Google Inc.
// http://code.google.com/p/google-diff-match-patch/

package diffmatchpatch

import (
	"bytes"
	"errors"
	"fmt"
	"html"
	"math"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// Operation defines the operation of a diff item.
type Operation int8

const (
	// DiffDelete item represents a delete diff.
	DiffDelete Operation = -1
	// DiffInsert item represents an insert diff.
	DiffInsert Operation = 1
	// DiffEqual item represents an equal diff.
	DiffEqual Operation = 0
)

// Diff represents one diff operation
type Diff struct {
	Type Operation
	Text string
}

func splice(slice []Diff, index int, amount int, elements ...Diff) []Diff {
	return append(slice[:index], append(elements, slice[index+amount:]...)...)
}

// DiffMain finds the differences between two texts.
// If an invalid UTF-8 sequence is encountered, it will be replaced by the Unicode replacement character.
func (dmp *DiffMatchPatch) DiffMain(text1, text2 string, checklines bool) []Diff {
	return dmp.DiffMainRunes([]rune(text1), []rune(text2), checklines)
}

// DiffMainRunes finds the differences between two rune sequences.
// If an invalid UTF-8 sequence is encountered, it will be replaced by the Unicode replacement character.
func (dmp *DiffMatchPatch) DiffMainRunes(text1, text2 []rune, checklines bool) []Diff {
	var deadline time.Time
	if dmp.DiffTimeout > 0 {
		deadline = time.Now().Add(dmp.DiffTimeout)
	}
	return dmp.diffMainRunes(text1, text2, checklines, deadline)
}

func (dmp *DiffMatchPatch) diffMainRunes(text1, text2 []rune, checklines bool, deadline time.Time) []Diff {
	if runesEqual(text1, text2) {
		var diffs []Diff
		if len(text1) > 0 {
			diffs = append(diffs, Diff{DiffEqual, string(text1)})
		}
		return diffs
	}
	// Trim off common prefix (speedup).
	commonlength := commonPrefixLength(text1, text2)
	commonprefix := text1[:commonlength]
	text1 = text1[commonlength:]
	text2 = text2[commonlength:]

	// Trim off common suffix (speedup).
	commonlength = commonSuffixLength(text1, text2)
	commonsuffix := text1[len(text1)-commonlength:]
	text1 = text1[:len(text1)-commonlength]
	text2 = text2[:len(text2)-commonlength]

	// Compute the diff on the middle block.
	diffs := dmp.diffCompute(text1, text2, checklines, deadline)

	// Restore the prefix and suffix.
	if len(commonprefix) != 0 {
		diffs = append([]Diff{Diff{DiffEqual, string(commonprefix)}}, diffs...)
	}
	if len(commonsuffix) != 0 {
		diffs = append(diffs, Diff{DiffEqual, string(commonsuffix)})
	}

	return dmp.DiffCleanupMerge(diffs)
}

// diffCompute finds the differences between two rune slices.  Assumes that the texts do not have any common prefix or suffix.
func (dmp *DiffMatchPatch) diffCompute(text1, text2 []rune, checklines bool, deadline time.Time) []Diff {
	diffs := []Diff{}
	if len(text1) == 0 {
		// Just add some text (speedup).
		return append(diffs, Diff{DiffInsert, string(text2)})
	} else if len(text2) == 0 {
		// Just delete some text (speedup).
		return append(diffs, Diff{DiffDelete, string(text1)})
	}

	var longtext, shorttext []rune
	if len(text1) > len(text2) {
		longtext = text1
		shorttext = text2
	} else {
		longtext = text2
		shorttext = text1
	}

	if i := runesIndex(longtext, shorttext); i != -1 {
		op := DiffInsert
		// Swap insertions for deletions if diff is reversed.
		if len(text1) > len(text2) {
			op = DiffDelete
		}
		// Shorter text is inside the longer text (speedup).
		return []Diff{
			Diff{op, string(longtext[:i])},
			Diff{DiffEqual, string(shorttext)},
			Diff{op, string(longtext[i+len(shorttext):])},
		}
	} else if len(shorttext) == 1 {
		// Single character string.
		// After the previous speedup, the character can't be an equality.
		return []Diff{
			Diff{DiffDelete, string(text1)},
			Diff{DiffInsert, string(text2)},
		}
		// Check to see if the problem can be split in two.
	} else if hm := dmp.diffHalfMatch(text1, text2); hm != nil {
		// A half-match was found, sort out the return data.
		text1A := hm[0]
		text1B := hm[1]
		text2A := hm[2]
		text2B := hm[3]
		midCommon := hm[4]
		// Send both pairs off for separate processing.
		diffsA := dmp.diffMainRunes(text1A, text2A, checklines, deadline)
		diffsB := dmp.diffMainRunes(text1B, text2B, checklines, deadline)
		// Merge the results.
		return append(diffsA, append([]Diff{Diff{DiffEqual, string(midCommon)}}, diffsB...)...)
	} else if checklines && len(text1) > 100 && len(text2) > 100 {
		return dmp.diffLineMode(text1, text2, deadline)
	}
	return dmp.diffBisect(text1, text2, deadline)
}

// diffLineMode does a quick line-level diff on both []runes, then rediff the parts for greater accuracy. This speedup can produce non-minimal diffs.
func (dmp *DiffMatchPatch) diffLineMode(text1, text2 []rune, deadline time.Time) []Diff {
	// Scan the text on a line-by-line basis first.
	text1, text2, linearray := dmp.diffLinesToRunes(text1, text2)

	diffs := dmp.diffMainRunes(text1, text2, false, deadline)

	// Convert the diff back to original text.
	diffs = dmp.DiffCharsToLines(diffs, linearray)
	// Eliminate freak matches (e.g. blank lines)
	diffs = dmp.DiffCleanupSemantic(diffs)

	// Rediff any replacement blocks, this time character-by-character.
	// Add a dummy entry at the end.
	diffs = append(diffs, Diff{DiffEqual, ""})

	pointer := 0
	countDelete := 0
	countInsert := 0

	// NOTE: Rune slices are slower than using strings in this case.
	textDelete := ""
	textInsert := ""

	for pointer < len(diffs) {
		switch diffs[pointer].Type {
		case DiffInsert:
			countInsert++
			textInsert += diffs[pointer].Text
		case DiffDelete:
			countDelete++
			textDelete += diffs[pointer].Text
		case DiffEqual:
			// Upon reaching an equality, check for prior redundancies.
			if countDelete >= 1 && countInsert >= 1 {
				// Delete the offending records and add the merged ones.
				diffs = splice(diffs, pointer-countDelete-countInsert,
					countDelete+countInsert)

				pointer = pointer - countDelete - countInsert
				a := dmp.diffMainRunes([]rune(textDelete), []rune(textInsert), false, deadline)
				for j := len(a) - 1; j >= 0; j-- {
					diffs = splice(diffs, pointer, 0, a[j])
				}
				pointer = pointer + len(a)
			}

			countInsert = 0
			countDelete = 0
			textDelete = ""
			textInsert = ""
		}
		pointer++
	}

	return diffs[:len(diffs)-1] // Remove the dummy entry at the end.
}

// DiffBisect finds the 'middle snake' of a diff, split the problem in two and return the recursively constructed diff.
// If an invalid UTF-8 sequence is encountered, it will be replaced by the Unicode replacement character.
// See Myers 1986 paper: An O(ND) Difference Algorithm and Its Variations.
func (dmp *DiffMatchPatch) DiffBisect(text1, text2 string, deadline time.Time) []Diff {
	// Unused in this code, but retained for interface compatibility.
	return dmp.diffBisect([]rune(text1), []rune(text2), deadline)
}

// diffBisect finds the 'middle snake' of a diff, splits the problem in two and returns the recursively constructed diff.
// See Myers's 1986 paper: An O(ND) Difference Algorithm and Its Variations.
func (dmp *DiffMatchPatch) diffBisect(runes1, runes2 []rune, deadline time.Time) []Diff {
	// Cache the text lengths to prevent multiple calls.
	runes1Len, runes2Len := len(runes1), len(runes2)

	maxD := (runes1Len + runes2Len + 1) / 2
	vOffset := maxD
	vLength := 2 * maxD

	v1 := make([]int, vLength)
	v2 := make([]int, vLength)
	for i := range v1 {
		v1[i] = -1
		v2[i] = -1
	}
	v1[vOffset+1] = 0
	v2[vOffset+1] = 0

	delta := runes1Len - runes2Len
	// If the total number of characters is odd, then the front path will collide with the reverse path.
	front := (delta%2 != 0)
	// Offsets for start and end of k loop. Prevents mapping of space beyond the grid.
	k1start := 0
	k1end := 0
	k2start := 0
	k2end := 0
	for d := 0; d < maxD; d++ {
		// Bail out if deadline is reached.
		if !deadline.IsZero() && time.Now().After(deadline) {
			break
		}

		// Walk the front path one step.
		for k1 := -d + k1start; k1 <= d-k1end; k1 += 2 {
			k1Offset := vOffset + k1
			var x1 int

			if k1 == -d || (k1 != d && v1[k1Offset-1] < v1[k1Offset+1]) {
				x1 = v1[k1Offset+1]
			} else {
				x1 = v1[k1Offset-1] + 1
			}

			y1 := x1 - k1
			for x1 < runes1Len && y1 < runes2Len {
				if runes1[x1] != runes2[y1] {
					break
				}
				x1++
				y1++
			}
			v1[k1Offset] = x1
			if x1 > runes1Len {
				// Ran off the right of the graph.
				k1end += 2
			} else if y1 > runes2Len {
				// Ran off the bottom of the graph.
				k1start += 2
			} else if front {
				k2Offset := vOffset + delta - k1
				if k2Offset >= 0 && k2Offset < vLength && v2[k2Offset] != -1 {
					// Mirror x2 onto top-left coordinate system.
					x2 := runes1Len - v2[k2Offset]
					if x1 >= x2 {
						// Overlap detected.
						return dmp.diffBisectSplit(runes1, runes2, x1, y1, deadline)
					}
				}
			}
		}
		// Walk the reverse path one step.
		for k2 := -d + k2start; k2 <= d-k2end; k2 += 2 {
			k2Offset := vOffset + k2
			var x2 int
			if k2 == -d || (k2 != d && v2[k2Offset-1] < v2[k2Offset+1]) {
				x2 = v2[k2Offset+1]
			} else {
				x2 = v2[k2Offset-1] + 1
			}
			var y2 = x2 - k2
			for x2 < runes1Len && y2 < runes2Len {
				if runes1[runes1Len-x2-1] != runes2[runes2Len-y2-1] {
					break
				}
				x2++
				y2++
			}
			v2[k2Offset] = x2
			if x2 > runes1Len {
				// Ran off the left of the graph.
				k2end += 2
			} else if y2 > runes2Len {
				// Ran off the top of the graph.
				k2start += 2
			} else if !front {
				k1Offset := vOffset + delta - k2
				if k1Offset >= 0 && k1Offset < vLength && v1[k1Offset] != -1 {
					x1 := v1[k1Offset]
					y1 := vOffset + x1 - k1Offset
					// Mirror x2 onto top-left coordinate system.
					x2 = runes1Len - x2
					if x1 >= x2 {
						// Overlap detected.
						return dmp.diffBisectSplit(runes1, runes2, x1, y1, deadline)
					}
				}
			}
		}
	}
	// Diff took too long and hit the deadline or number of diffs equals number of characters, no commonality at all.
	return []Diff{
		Diff{DiffDelete, string(runes1)},
		Diff{DiffInsert, string(runes2)},
	}
}

func (dmp *DiffMatchPatch) diffBisectSplit(runes1, runes2 []rune, x, y int,
	deadline time.Time) []Diff {
	runes1a := runes1[:x]
	runes2a := runes2[:y]
	runes1b := runes1[x:]
	runes2b := runes2[y:]

	// Compute both diffs serially.
	diffs := dmp.diffMainRunes(runes1a, runes2a, false, deadline)
	diffsb := dmp.diffMainRunes(runes1b, runes2b, false, deadline)

	return append(diffs, diffsb...)
}

// DiffLinesToChars splits two texts into a list of strings, and educes the texts to a string of hashes where each Unicode character represents one line.
// It's slightly faster to call DiffLinesToRunes first, followed by DiffMainRunes.
func (dmp *DiffMatchPatch) DiffLinesToChars(text1, text2 string) (string, string, []string) {
	chars1, chars2, lineArray := dmp.DiffLinesToRunes(text1, text2)
	return string(chars1), string(chars2), lineArray
}

// DiffLinesToRunes splits two texts into a list of runes. Each rune represents one line.
func (dmp *DiffMatchPatch) DiffLinesToRunes(text1, text2 string) ([]rune, []rune, []string) {
	// '\x00' is a valid character, but various debuggers don't like it. So we'll insert a junk entry to avoid generating a null character.
	lineArray := []string{""}    // e.g. lineArray[4] == 'Hello\n'
	lineHash := map[string]int{} // e.g. lineHash['Hello\n'] == 4

	chars1 := dmp.diffLinesToRunesMunge(text1, &lineArray, lineHash)
	chars2 := dmp.diffLinesToRunesMunge(text2, &lineArray, lineHash)

	return chars1, chars2, lineArray
}

func (dmp *DiffMatchPatch) diffLinesToRunes(text1, text2 []rune) ([]rune, []rune, []string) {
	return dmp.DiffLinesToRunes(string(text1), string(text2))
}

// diffLinesToRunesMunge splits a text into an array of strings, and reduces the texts to a []rune where each Unicode character represents one line.
// We use strings instead of []runes as input mainly because you can't use []rune as a map key.
func (dmp *DiffMatchPatch) diffLinesToRunesMunge(text string, lineArray *[]string, lineHash map[string]int) []rune {
	// Walk the text, pulling out a substring for each line. text.split('\n') would would temporarily double our memory footprint. Modifying text would create many large strings to garbage collect.
	lineStart := 0
	lineEnd := -1
	runes := []rune{}

	for lineEnd < len(text)-1 {
		lineEnd = indexOf(text, "\n", lineStart)

		if lineEnd == -1 {
			lineEnd = len(text) - 1
		}

		line := text[lineStart : lineEnd+1]
		lineStart = lineEnd + 1
		lineValue, ok := lineHash[line]

		if ok {
			runes = append(runes, rune(lineValue))
		} else {
			*lineArray = append(*lineArray, line)
			lineHash[line] = len(*lineArray) - 1
			runes = append(runes, rune(len(*lineArray)-1))
		}
	}

	return runes
}

// DiffCharsToLines rehydrates the text in a diff from a string of line hashes to real lines of text.
func (dmp *DiffMatchPatch) DiffCharsToLines(diffs []Diff, lineArray []string) []Diff {
	hydrated := make([]Diff, 0, len(diffs))
	for _, aDiff := range diffs {
		chars := aDiff.Text
		text := make([]string, len(chars))

		for i, r := range chars {
			text[i] = lineArray[r]
		}

		aDiff.Text = strings.Join(text, "")
		hydrated = append(hydrated, aDiff)
	}
	return hydrated
}

// DiffCommonPrefix determines the common prefix length of two strings.
func (dmp *DiffMatchPatch) DiffCommonPrefix(text1, text2 string) int {
	// Unused in this code, but retained for interface compatibility.
	return commonPrefixLength([]rune(text1), []rune(text2))
}

// DiffCommonSuffix determines the common suffix length of two strings.
func (dmp *DiffMatchPatch) DiffCommonSuffix(text1, text2 string) int {
	// Unused in this code, but retained for interface compatibility.
	return commonSuffixLength([]rune(text1), []rune(text2))
}

// commonPrefixLength returns the length of the common prefix of two rune slices.
func commonPrefixLength(text1, text2 []rune) int {
	short, long := text1, text2
	if len(short) > len(long) {
		short, long = long, short
	}
	for i, r := range short {
		if r != long[i] {
			return i
		}
	}
	return len(short)
}

// commonSuffixLength returns the length of the common suffix of two rune slices.
func commonSuffixLength(text1, text2 []rune) int {
	n := min(len(text1), len(text2))
	for i := 0; i < n; i++ {
		if text1[len(text1)-i-1] != text2[len(text2)-i-1] {
			return i
		}
	}
	return n

	// TODO research and benchmark this, why is it not activated? https://github.com/sergi/go-diff/issues/54
	// Binary search.
	// Performance analysis: http://neil.fraser.name/news/2007/10/09/
	/*
	   pointermin := 0
	   pointermax := math.Min(len(text1), len(text2))
	   pointermid := pointermax
	   pointerend := 0
	   for pointermin < pointermid {
	       if text1[len(text1)-pointermid:len(text1)-pointerend] ==
	           text2[len(text2)-pointermid:len(text2)-pointerend] {
	           pointermin = pointermid
	           pointerend = pointermin
	       } else {
	           pointermax = pointermid
	       }
	       pointermid = math.Floor((pointermax-pointermin)/2 + pointermin)
	   }
	   return pointermid
	*/
}

// DiffCommonOverlap determines if the suffix of one string is the prefix of another.
func (dmp *DiffMatchPatch) DiffCommonOverlap(text1 string, text2 string) int {
	// Cache the text lengths to prevent multiple calls.
	text1Length := len(text1)
	text2Length := len(text2)
	// Eliminate the null case.
	if text1Length == 0 || text2Length == 0 {
		return 0
	}
	// Truncate the longer string.
	if text1Length > text2Length {
		text1 = text1[text1Length-text2Length:]
	} else if text1Length < text2Length {
		text2 = text2[0:text1Length]
	}
	textLength := int(math.Min(float64(text1Length), float64(text2Length)))
	// Quick check for the worst case.
	if text1 == text2 {
		return textLength
	}

	// Start by looking for a single character match and increase length until no match is found. Performance analysis: http://neil.fraser.name/news/2010/11/04/
	best := 0
	length := 1
	for {
		pattern := text1[textLength-length:]
		found := strings.Index(text2, pattern)
		if found == -1 {
			break
		}
		length += found
		if found == 0 || text1[textLength-length:] == text2[0:length] {
			best = length
			length++
		}
	}

	return best
}

// DiffHalfMatch checks whether the two texts share a substring which is at least half the length of the longer text. This speedup can produce non-minimal diffs.
func (dmp *DiffMatchPatch) DiffHalfMatch(text1, text2 string) []string {
	// Unused in this code, but retained for interface compatibility.
	runeSlices := dmp.diffHalfMatch([]rune(text1), []rune(text2))
	if runeSlices == nil {
		return nil
	}

	result := make([]string, len(runeSlices))
	for i, r := range runeSlices {
		result[i] = string(r)
	}
	return result
}

func (dmp *DiffMatchPatch) diffHalfMatch(text1, text2 []rune) [][]rune {
	if dmp.DiffTimeout <= 0 {
		// Don't risk returning a non-optimal diff if we have unlimited time.
		return nil
	}

	var longtext, shorttext []rune
	if len(text1) > len(text2) {
		longtext = text1
		shorttext = text2
	} else {
		longtext = text2
		shorttext = text1
	}

	if len(longtext) < 4 || len(shorttext)*2 < len(longtext) {
		return nil // Pointless.
	}

	// First check if the second quarter is the seed for a half-match.
	hm1 := dmp.diffHalfMatchI(longtext, shorttext, int(float64(len(longtext)+3)/4))

	// Check again based on the third quarter.
	hm2 := dmp.diffHalfMatchI(longtext, shorttext, int(float64(len(longtext)+1)/2))

	hm := [][]rune{}
	if hm1 == nil && hm2 == nil {
		return nil
	} else if hm2 == nil {
		hm = hm1
	} else if hm1 == nil {
		hm = hm2
	} else {
		// Both matched.  Select the longest.
		if len(hm1[4]) > len(hm2[4]) {
			hm = hm1
		} else {
			hm = hm2
		}
	}

	// A half-match was found, sort out the return data.
	if len(text1) > len(text2) {
		return hm
	}

	return [][]rune{hm[2], hm[3], hm[0], hm[1], hm[4]}
}

// diffHalfMatchI checks if a substring of shorttext exist within longtext such that the substring is at least half the length of longtext?
// Returns a slice containing the prefix of longtext, the suffix of longtext, the prefix of shorttext, the suffix of shorttext and the common middle, or null if there was no match.
func (dmp *DiffMatchPatch) diffHalfMatchI(l, s []rune, i int) [][]rune {
	var bestCommonA []rune
	var bestCommonB []rune
	var bestCommonLen int
	var bestLongtextA []rune
	var bestLongtextB []rune
	var bestShorttextA []rune
	var bestShorttextB []rune

	// Start with a 1/4 length substring at position i as a seed.
	seed := l[i : i+len(l)/4]

	for j := runesIndexOf(s, seed, 0); j != -1; j = runesIndexOf(s, seed, j+1) {
		prefixLength := commonPrefixLength(l[i:], s[j:])
		suffixLength := commonSuffixLength(l[:i], s[:j])

		if bestCommonLen < suffixLength+prefixLength {
			bestCommonA = s[j-suffixLength : j]
			bestCommonB = s[j : j+prefixLength]
			bestCommonLen = len(bestCommonA) + len(bestCommonB)
			bestLongtextA = l[:i-suffixLength]
			bestLongtextB = l[i+prefixLength:]
			bestShorttextA = s[:j-suffixLength]
			bestShorttextB = s[j+prefixLength:]
		}
	}

	if bestCommonLen*2 < len(l) {
		return nil
	}

	return [][]rune{
		bestLongtextA,
		bestLongtextB,
		bestShorttextA,
		bestShorttextB,
		append(bestCommonA, bestCommonB...),
	}
}

// DiffCleanupSemantic reduces the number of edits by eliminating semantically trivial equalities.
func (dmp *DiffMatchPatch) DiffCleanupSemantic(diffs []Diff) []Diff {
	changes := false
	// Stack of indices where equalities are found.
	type equality struct {
		data int
		next *equality
	}
	var equalities *equality

	var lastequality string
	// Always equal to diffs[equalities[equalitiesLength - 1]][1]
	var pointer int // Index of current position.
	// Number of characters that changed prior to the equality.
	var lengthInsertions1, lengthDeletions1 int
	// Number of characters that changed after the equality.
	var lengthInsertions2, lengthDeletions2 int

	for pointer < len(diffs) {
		if diffs[pointer].Type == DiffEqual {
			// Equality found.

			equalities = &equality{
				data: pointer,
				next: equalities,
			}
			lengthInsertions1 = lengthInsertions2
			lengthDeletions1 = lengthDeletions2
			lengthInsertions2 = 0
			lengthDeletions2 = 0
			lastequality = diffs[pointer].Text
		} else {
			// An insertion or deletion.

			if diffs[pointer].Type == DiffInsert {
				lengthInsertions2 += len(diffs[pointer].Text)
			} else {
				lengthDeletions2 += len(diffs[pointer].Text)
			}
			// Eliminate an equality that is smaller or equal to the edits on both sides of it.
			difference1 := int(math.Max(float64(lengthInsertions1), float64(lengthDeletions1)))
			difference2 := int(math.Max(float64(lengthInsertions2), float64(lengthDeletions2)))
			if len(lastequality) > 0 &&
				(len(lastequality) <= difference1) &&
				(len(lastequality) <= difference2) {
				// Duplicate record.
				insPoint := equalities.data
				diffs = append(
					diffs[:insPoint],
					append([]Diff{Diff{DiffDelete, lastequality}}, diffs[insPoint:]...)...)

				// Change second copy to insert.
				diffs[insPoint+1].Type = DiffInsert
				// Throw away the equality we just deleted.
				equalities = equalities.next

				if equalities != nil {
					equalities = equalities.next
				}
				if equalities != nil {
					pointer = equalities.data
				} else {
					pointer = -1
				}

				lengthInsertions1 = 0 // Reset the counters.
				lengthDeletions1 = 0
				lengthInsertions2 = 0
				lengthDeletions2 = 0
				lastequality = ""
				changes = true
			}
		}
		pointer++
	}

	// Normalize the diff.
	if changes {
		diffs = dmp.DiffCleanupMerge(diffs)
	}
	diffs = dmp.DiffCleanupSemanticLossless(diffs)
	// Find any overlaps between deletions and insertions.
	// e.g: <del>abcxxx</del><ins>xxxdef</ins>
	//   -> <del>abc</del>xxx<ins>def</ins>
	// e.g: <del>xxxabc</del><ins>defxxx</ins>
	//   -> <ins>def</ins>xxx<del>abc</del>
	// Only extract an overlap if it is as big as the edit ahead or behind it.
	pointer = 1
	for pointer < len(diffs) {
		if diffs[pointer-1].Type == DiffDelete &&
			diffs[pointer].Type == DiffInsert {
			deletion := diffs[pointer-1].Text
			insertion := diffs[pointer].Text
			overlapLength1 := dmp.DiffCommonOverlap(deletion, insertion)
			overlapLength2 := dmp.DiffCommonOverlap(insertion, deletion)
			if overlapLength1 >= overlapLength2 {
				if float64(overlapLength1) >= float64(len(deletion))/2 ||
					float64(overlapLength1) >= float64(len(insertion))/2 {

					// Overlap found. Insert an equality and trim the surrounding edits.
					diffs = append(
						diffs[:pointer],
						append([]Diff{Diff{DiffEqual, insertion[:overlapLength1]}}, diffs[pointer:]...)...)

					diffs[pointer-1].Text =
						deletion[0 : len(deletion)-overlapLength1]
					diffs[pointer+1].Text = insertion[overlapLength1:]
					pointer++
				}
			} else {
				if float64(overlapLength2) >= float64(len(deletion))/2 ||
					float64(overlapLength2) >= float64(len(insertion))/2 {
					// Reverse overlap found. Insert an equality and swap and trim the surrounding edits.
					overlap := Diff{DiffEqual, deletion[:overlapLength2]}
					diffs = append(
						diffs[:pointer],
						append([]Diff{overlap}, diffs[pointer:]...)...)

					diffs[pointer-1].Type = DiffInsert
					diffs[pointer-1].Text = insertion[0 : len(insertion)-overlapLength2]
					diffs[pointer+1].Type = DiffDelete
					diffs[pointer+1].Text = deletion[overlapLength2:]
					pointer++
				}
			}
			pointer++
		}
		pointer++
	}

	return diffs
}

// Define some regex patterns for matching boundaries.
var (
	nonAlphaNumericRegex = regexp.MustCompile(`[^a-zA-Z0-9]`)
	whitespaceRegex      = regexp.MustCompile(`\s`)
	linebreakRegex       = regexp.MustCompile(`[\r\n]`)
	blanklineEndRegex    = regexp.MustCompile(`\n\r?\n$`)
	blanklineStartRegex  = regexp.MustCompile(`^\r?\n\r?\n`)
)

// diffCleanupSemanticScore computes a score representing whether the internal boundary falls on logical boundaries.
// Scores range from 6 (best) to 0 (worst). Closure, but does not reference any external variables.
func diffCleanupSemanticScore(one, two string) int {
	if len(one) == 0 || len(two) == 0 {
		// Edges are the best.
		return 6
	}

	// Each port of this function behaves slightly differently due to subtle differences in each language's definition of things like 'whitespace'.  Since this function's purpose is largely cosmetic, the choice has been made to use each language's native features rather than force total conformity.
	rune1, _ := utf8.DecodeLastRuneInString(one)
	rune2, _ := utf8.DecodeRuneInString(two)
	char1 := string(rune1)
	char2 := string(rune2)

	nonAlphaNumeric1 := nonAlphaNumericRegex.MatchString(char1)
	nonAlphaNumeric2 := nonAlphaNumericRegex.MatchString(char2)
	whitespace1 := nonAlphaNumeric1 && whitespaceRegex.MatchString(char1)
	whitespace2 := nonAlphaNumeric2 && whitespaceRegex.MatchString(char2)
	lineBreak1 := whitespace1 && linebreakRegex.MatchString(char1)
	lineBreak2 := whitespace2 && linebreakRegex.MatchString(char2)
	blankLine1 := lineBreak1 && blanklineEndRegex.MatchString(one)
	blankLine2 := lineBreak2 && blanklineEndRegex.MatchString(two)

	if blankLine1 || blankLine2 {
		// Five points for blank lines.
		return 5
	} else if lineBreak1 || lineBreak2 {
		// Four points for line breaks.
		return 4
	} else if nonAlphaNumeric1 && !whitespace1 && whitespace2 {
		// Three points for end of sentences.
		return 3
	} else if whitespace1 || whitespace2 {
		// Two points for whitespace.
		return 2
	} else if nonAlphaNumeric1 || nonAlphaNumeric2 {
		// One point for non-alphanumeric.
		return 1
	}
	return 0
}

// DiffCleanupSemanticLossless looks for single edits surrounded on both sides by equalities which can be shifted sideways to align the edit to a word boundary.
// E.g: The c<ins>at c</ins>ame. -> The <ins>cat </ins>came.
func (dmp *DiffMatchPatch) DiffCleanupSemanticLossless(diffs []Diff) []Diff {
	pointer := 1

	// Intentionally ignore the first and last element (don't need checking).
	for pointer < len(diffs)-1 {
		if diffs[pointer-1].Type == DiffEqual &&
			diffs[pointer+1].Type == DiffEqual {

			// This is a single edit surrounded by equalities.
			equality1 := diffs[pointer-1].Text
			edit := diffs[pointer].Text
			equality2 := diffs[pointer+1].Text

			// First, shift the edit as far left as possible.
			commonOffset := dmp.DiffCommonSuffix(equality1, edit)
			if commonOffset > 0 {
				commonString := edit[len(edit)-commonOffset:]
				equality1 = equality1[0 : len(equality1)-commonOffset]
				edit = commonString + edit[:len(edit)-commonOffset]
				equality2 = commonString + equality2
			}

			// Second, step character by character right, looking for the best fit.
			bestEquality1 := equality1
			bestEdit := edit
			bestEquality2 := equality2
			bestScore := diffCleanupSemanticScore(equality1, edit) +
				diffCleanupSemanticScore(edit, equality2)

			for len(edit) != 0 && len(equality2) != 0 {
				_, sz := utf8.DecodeRuneInString(edit)
				if len(equality2) < sz || edit[:sz] != equality2[:sz] {
					break
				}
				equality1 += edit[:sz]
				edit = edit[sz:] + equality2[:sz]
				equality2 = equality2[sz:]
				score := diffCleanupSemanticScore(equality1, edit) +
					diffCleanupSemanticScore(edit, equality2)
				// The >= encourages trailing rather than leading whitespace on edits.
				if score >= bestScore {
					bestScore = score
					bestEquality1 = equality1
					bestEdit = edit
					bestEquality2 = equality2
				}
			}

			if diffs[pointer-1].Text != bestEquality1 {
				// We have an improvement, save it back to the diff.
				if len(bestEquality1) != 0 {
					diffs[pointer-1].Text = bestEquality1
				} else {
					diffs = splice(diffs, pointer-1, 1)
					pointer--
				}

				diffs[pointer].Text = bestEdit
				if len(bestEquality2) != 0 {
					diffs[pointer+1].Text = bestEquality2
				} else {
					diffs = append(diffs[:pointer+1], diffs[pointer+2:]...)
					pointer--
				}
			}
		}
		pointer++
	}

	return diffs
}

// DiffCleanupEfficiency reduces the number of edits by eliminating operationally trivial equalities.
func (dmp *DiffMatchPatch) DiffCleanupEfficiency(diffs []Diff) []Diff {
	changes := false
	// Stack of indices where equalities are found.
	type equality struct {
		data int
		next *equality
	}
	var equalities *equality
	// Always equal to equalities[equalitiesLength-1][1]
	lastequality := ""
	pointer := 0 // Index of current position.
	// Is there an insertion operation before the last equality.
	preIns := false
	// Is there a deletion operation before the last equality.
	preDel := false
	// Is there an insertion operation after the last equality.
	postIns := false
	// Is there a deletion operation after the last equality.
	postDel := false
	for pointer < len(diffs) {
		if diffs[pointer].Type == DiffEqual { // Equality found.
			if len(diffs[pointer].Text) < dmp.DiffEditCost &&
				(postIns || postDel) {
				// Candidate found.
				equalities = &equality{
					data: pointer,
					next: equalities,
				}
				preIns = postIns
				preDel = postDel
				lastequality = diffs[pointer].Text
			} else {
				// Not a candidate, and can never become one.
				equalities = nil
				lastequality = ""
			}
			postIns = false
			postDel = false
		} else { // An insertion or deletion.
			if diffs[pointer].Type == DiffDelete {
				postDel = true
			} else {
				postIns = true
			}

			// Five types to be split:
			// <ins>A</ins><del>B</del>XY<ins>C</ins><del>D</del>
			// <ins>A</ins>X<ins>C</ins><del>D</del>
			// <ins>A</ins><del>B</del>X<ins>C</ins>
			// <ins>A</del>X<ins>C</ins><del>D</del>
			// <ins>A</ins><del>B</del>X<del>C</del>
			var sumPres int
			if preIns {
				sumPres++
			}
			if preDel {
				sumPres++
			}
			if postIns {
				sumPres++
			}
			if postDel {
				sumPres++
			}
			if len(lastequality) > 0 &&
				((preIns && preDel && postIns && postDel) ||
					((len(lastequality) < dmp.DiffEditCost/2) && sumPres == 3)) {

				insPoint := equalities.data

				// Duplicate record.
				diffs = append(diffs[:insPoint],
					append([]Diff{Diff{DiffDelete, lastequality}}, diffs[insPoint:]...)...)

				// Change second copy to insert.
				diffs[insPoint+1].Type = DiffInsert
				// Throw away the equality we just deleted.
				equalities = equalities.next
				lastequality = ""

				if preIns && preDel {
					// No changes made which could affect previous entry, keep going.
					postIns = true
					postDel = true
					equalities = nil
				} else {
					if equalities != nil {
						equalities = equalities.next
					}
					if equalities != nil {
						pointer = equalities.data
					} else {
						pointer = -1
					}
					postIns = false
					postDel = false
				}
				changes = true
			}
		}
		pointer++
	}

	if changes {
		diffs = dmp.DiffCleanupMerge(diffs)
	}

	return diffs
}

// DiffCleanupMerge reorders and merges like edit sections. Merge equalities.
// Any edit section can move as long as it doesn't cross an equality.
func (dmp *DiffMatchPatch) DiffCleanupMerge(diffs []Diff) []Diff {
	// Add a dummy entry at the end.
	diffs = append(diffs, Diff{DiffEqual, ""})
	pointer := 0
	countDelete := 0
	countInsert := 0
	commonlength := 0
	textDelete := []rune(nil)
	textInsert := []rune(nil)

	for pointer < len(diffs) {
		switch diffs[pointer].Type {
		case DiffInsert:
			countInsert++
			textInsert = append(textInsert, []rune(diffs[pointer].Text)...)
			pointer++
			break
		case DiffDelete:
			countDelete++
			textDelete = append(textDelete, []rune(diffs[pointer].Text)...)
			pointer++
			break
		case DiffEqual:
			// Upon reaching an equality, check for prior redundancies.
			if countDelete+countInsert > 1 {
				if countDelete != 0 && countInsert != 0 {
					// Factor out any common prefixies.
					commonlength = commonPrefixLength(textInsert, textDelete)
					if commonlength != 0 {
						x := pointer - countDelete - countInsert
						if x > 0 && diffs[x-1].Type == DiffEqual {
							diffs[x-1].Text += string(textInsert[:commonlength])
						} else {
							diffs = append([]Diff{Diff{DiffEqual, string(textInsert[:commonlength])}}, diffs...)
							pointer++
						}
						textInsert = textInsert[commonlength:]
						textDelete = textDelete[commonlength:]
					}
					// Factor out any common suffixies.
					commonlength = commonSuffixLength(textInsert, textDelete)
					if commonlength != 0 {
						insertIndex := len(textInsert) - commonlength
						deleteIndex := len(textDelete) - commonlength
						diffs[pointer].Text = string(textInsert[insertIndex:]) + diffs[pointer].Text
						textInsert = textInsert[:insertIndex]
						textDelete = textDelete[:deleteIndex]
					}
				}
				// Delete the offending records and add the merged ones.
				if countDelete == 0 {
					diffs = splice(diffs, pointer-countInsert,
						countDelete+countInsert,
						Diff{DiffInsert, string(textInsert)})
				} else if countInsert == 0 {
					diffs = splice(diffs, pointer-countDelete,
						countDelete+countInsert,
						Diff{DiffDelete, string(textDelete)})
				} else {
					diffs = splice(diffs, pointer-countDelete-countInsert,
						countDelete+countInsert,
						Diff{DiffDelete, string(textDelete)},
						Diff{DiffInsert, string(textInsert)})
				}

				pointer = pointer - countDelete - countInsert + 1
				if countDelete != 0 {
					pointer++
				}
				if countInsert != 0 {
					pointer++
				}
			} else if pointer != 0 && diffs[pointer-1].Type == DiffEqual {
				// Merge this equality with the previous one.
				diffs[pointer-1].Text += diffs[pointer].Text
				diffs = append(diffs[:pointer], diffs[pointer+1:]...)
			} else {
				pointer++
			}
			countInsert = 0
			countDelete = 0
			textDelete = nil
			textInsert = nil
			break
		}
	}

	if len(diffs[len(diffs)-1].Text) == 0 {
		diffs = diffs[0 : len(diffs)-1] // Remove the dummy entry at the end.
	}

	// Second pass: look for single edits surrounded on both sides by equalities which can be shifted sideways to eliminate an equality. E.g: A<ins>BA</ins>C -> <ins>AB</ins>AC
	changes := false
	pointer = 1
	// Intentionally ignore the first and last element (don't need checking).
	for pointer < (len(diffs) - 1) {
		if diffs[pointer-1].Type == DiffEqual &&
			diffs[pointer+1].Type == DiffEqual {
			// This is a single edit surrounded by equalities.
			if strings.HasSuffix(diffs[pointer].Text, diffs[pointer-1].Text) {
				// Shift the edit over the previous equality.
				diffs[pointer].Text = diffs[pointer-1].Text +
					diffs[pointer].Text[:len(diffs[pointer].Text)-len(diffs[pointer-1].Text)]
				diffs[pointer+1].Text = diffs[pointer-1].Text + diffs[pointer+1].Text
				diffs = splice(diffs, pointer-1, 1)
				changes = true
			} else if strings.HasPrefix(diffs[pointer].Text, diffs[pointer+1].Text) {
				// Shift the edit over the next equality.
				diffs[pointer-1].Text += diffs[pointer+1].Text
				diffs[pointer].Text =
					diffs[pointer].Text[len(diffs[pointer+1].Text):] + diffs[pointer+1].Text
				diffs = splice(diffs, pointer+1, 1)
				changes = true
			}
		}
		pointer++
	}

	// If shifts were made, the diff needs reordering and another shift sweep.
	if changes {
		diffs = dmp.DiffCleanupMerge(diffs)
	}

	return diffs
}

// DiffXIndex returns the equivalent location in s2.
func (dmp *DiffMatchPatch) DiffXIndex(diffs []Diff, loc int) int {
	chars1 := 0
	chars2 := 0
	lastChars1 := 0
	lastChars2 := 0
	lastDiff := Diff{}
	for i := 0; i < len(diffs); i++ {
		aDiff := diffs[i]
		if aDiff.Type != DiffInsert {
			// Equality or deletion.
			chars1 += len(aDiff.Text)
		}
		if aDiff.Type != DiffDelete {
			// Equality or insertion.
			chars2 += len(aDiff.Text)
		}
		if chars1 > loc {
			// Overshot the location.
			lastDiff = aDiff
			break
		}
		lastChars1 = chars1
		lastChars2 = chars2
	}
	if lastDiff.Type == DiffDelete {
		// The location was deleted.
		return lastChars2
	}
	// Add the remaining character length.
	return lastChars2 + (loc - lastChars1)
}

// DiffPrettyHtml converts a []Diff into a pretty HTML report.
// It is intended as an example from which to write one's own display functions.
func (dmp *DiffMatchPatch) DiffPrettyHtml(diffs []Diff) string {
	var buff bytes.Buffer
	for _, diff := range diffs {
		text := strings.Replace(html.EscapeString(diff.Text), "\n", "&para;<br>", -1)
		switch diff.Type {
		case DiffInsert:
			_, _ = buff.WriteString("<ins style=\"background:#e6ffe6;\">")
			_, _ = buff.WriteString(text)
			_, _ = buff.WriteString("</ins>")
		case DiffDelete:
			_, _ = buff.WriteString("<del style=\"background:#ffe6e6;\">")
			_, _ = buff.WriteString(text)
			_, _ = buff.WriteString("</del>")
		case DiffEqual:
			_, _ = buff.WriteString("<span>")
			_, _ = buff.WriteString(text)
			_, _ = buff.WriteString("</span>")
		}
	}
	return buff.String()
}

// DiffPrettyText converts a []Diff into a colored text report.
func (dmp *DiffMatchPatch) DiffPrettyText(diffs []Diff) string {
	var buff bytes.Buffer
	for _, diff := range diffs {
		text := diff.Text

		switch diff.Type {
		case DiffInsert:
			_, _ = buff.WriteString("\x1b[32m")
			_, _ = buff.WriteString(text)
			_, _ = buff.WriteString("\x1b[0m")
		case DiffDelete:
			_, _ = buff.WriteString("\x1b[31m")
			_, _ = buff.WriteString(text)
			_, _ = buff.WriteString("\x1b[0m")
		case DiffEqual:
			_, _ = buff.WriteString(text)
		}
	}

	return buff.String()
}

// DiffText1 computes and returns the source text (all equalities and deletions).
func (dmp *DiffMatchPatch) DiffText1(diffs []Diff) string {
	//StringBuilder text = new StringBuilder()
	var text bytes.Buffer

	for _, aDiff := range diffs {
		if aDiff.Type != DiffInsert {
			_, _ = text.WriteString(aDiff.Text)
		}
	}
	return text.String()
}

// DiffText2 computes and returns the destination text (all equalities and insertions).
func (dmp *DiffMatchPatch) DiffText2(diffs []Diff) string {
	var text bytes.Buffer

	for _, aDiff := range diffs {
		if aDiff.Type != DiffDelete {
			_, _ = text.WriteString(aDiff.Text)
		}
	}
	return text.String()
}

// DiffLevenshtein computes the Levenshtein distance that is the number of inserted, deleted or substituted characters.
func (dmp *DiffMatchPatch) DiffLevenshtein(diffs []Diff) int {
	levenshtein := 0
	insertions := 0
	deletions := 0

	for _, aDiff := range diffs {
		switch aDiff.Type {
		case DiffInsert:
			insertions += len(aDiff.Text)
		case DiffDelete:
			deletions += len(aDiff.Text)
		case DiffEqual:
			// A deletion and an insertion is one substitution.
			levenshtein += max(insertions, deletions)
			insertions = 0
			deletions = 0
		}
	}

	levenshtein += max(insertions, deletions)
	return levenshtein
}

// DiffToDelta crushes the diff into an encoded string which describes the operations required to transform text1 into text2.
// E.g. =3\t-2\t+ing  -> Keep 3 chars, delete 2 chars, insert 'ing'. Operations are tab-separated.  Inserted text is escaped using %xx notation.
func (dmp *DiffMatchPatch) DiffToDelta(diffs []Diff) string {
	var text bytes.Buffer
	for _, aDiff := range diffs {
		switch aDiff.Type {
		case DiffInsert:
			_, _ = text.WriteString("+")
			_, _ = text.WriteString(strings.Replace(url.QueryEscape(aDiff.Text), "+", " ", -1))
			_, _ = text.WriteString("\t")
			break
		case DiffDelete:
			_, _ = text.WriteString("-")
			_, _ = text.WriteString(strconv.Itoa(utf8.RuneCountInString(aDiff.Text)))
			_, _ = text.WriteString("\t")
			break
		case DiffEqual:
			_, _ = text.WriteString("=")
			_, _ = text.WriteString(strconv.Itoa(utf8.RuneCountInString(aDiff.Text)))
			_, _ = text.WriteString("\t")
			break
		}
	}
	delta := text.String()
	if len(delta) != 0 {
		// Strip off trailing tab character.
		delta = delta[0 : utf8.RuneCountInString(delta)-1]
		delta = unescaper.Replace(delta)
	}
	return delta
}

// DiffFromDelta given the original text1, and an encoded string which describes the operations required to transform text1 into text2, comAdde the full diff.
func (dmp *DiffMatchPatch) DiffFromDelta(text1 string, delta string) (diffs []Diff, err error) {
	i := 0
	runes := []rune(text1)

	for _, token := range strings.Split(delta, "\t") {
		if len(token) == 0 {
			// Blank tokens are ok (from a trailing \t).
			continue
		}

		// Each token begins with a one character parameter which specifies the operation of this token (delete, insert, equality).
		param := token[1:]

		switch op := token[0]; op {
		case '+':
			// Decode would Diff all "+" to " "
			param = strings.Replace(param, "+", "%2b", -1)
			param, err = url.QueryUnescape(param)
			if err != nil {
				return nil, err
			}
			if !utf8.ValidString(param) {
				return nil, fmt.Errorf("invalid UTF-8 token: %q", param)
			}

			diffs = append(diffs, Diff{DiffInsert, param})
		case '=', '-':
			n, err := strconv.ParseInt(param, 10, 0)
			if err != nil {
				return nil, err
			} else if n < 0 {
				return nil, errors.New("Negative number in DiffFromDelta: " + param)
			}

			i += int(n)
			// Break out if we are out of bounds, go1.6 can't handle this very well
			if i > len(runes) {
				break
			}
			// Remember that string slicing is by byte - we want by rune here.
			text := string(runes[i-int(n) : i])

			if op == '=' {
				diffs = append(diffs, Diff{DiffEqual, text})
			} else {
				diffs = append(diffs, Diff{DiffDelete, text})
			}
		default:
			// Anything else is an error.
			return nil, errors.New("Invalid diff operation in DiffFromDelta: " + string(token[0]))
		}
	}

	if i != len(runes) {
		return nil, fmt.Errorf("Delta length (%v) is different from source text length (%v)", i, len(text1))
	}

	return diffs, nil
}
