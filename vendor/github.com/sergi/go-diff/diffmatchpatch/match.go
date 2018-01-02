// Copyright (c) 2012-2016 The go-diff authors. All rights reserved.
// https://github.com/sergi/go-diff
// See the included LICENSE file for license details.
//
// go-diff is a Go implementation of Google's Diff, Match, and Patch library
// Original library is Copyright (c) 2006 Google Inc.
// http://code.google.com/p/google-diff-match-patch/

package diffmatchpatch

import (
	"math"
)

// MatchMain locates the best instance of 'pattern' in 'text' near 'loc'.
// Returns -1 if no match found.
func (dmp *DiffMatchPatch) MatchMain(text, pattern string, loc int) int {
	// Check for null inputs not needed since null can't be passed in C#.

	loc = int(math.Max(0, math.Min(float64(loc), float64(len(text)))))
	if text == pattern {
		// Shortcut (potentially not guaranteed by the algorithm)
		return 0
	} else if len(text) == 0 {
		// Nothing to match.
		return -1
	} else if loc+len(pattern) <= len(text) && text[loc:loc+len(pattern)] == pattern {
		// Perfect match at the perfect spot!  (Includes case of null pattern)
		return loc
	}
	// Do a fuzzy compare.
	return dmp.MatchBitap(text, pattern, loc)
}

// MatchBitap locates the best instance of 'pattern' in 'text' near 'loc' using the Bitap algorithm.
// Returns -1 if no match was found.
func (dmp *DiffMatchPatch) MatchBitap(text, pattern string, loc int) int {
	// Initialise the alphabet.
	s := dmp.MatchAlphabet(pattern)

	// Highest score beyond which we give up.
	scoreThreshold := dmp.MatchThreshold
	// Is there a nearby exact match? (speedup)
	bestLoc := indexOf(text, pattern, loc)
	if bestLoc != -1 {
		scoreThreshold = math.Min(dmp.matchBitapScore(0, bestLoc, loc,
			pattern), scoreThreshold)
		// What about in the other direction? (speedup)
		bestLoc = lastIndexOf(text, pattern, loc+len(pattern))
		if bestLoc != -1 {
			scoreThreshold = math.Min(dmp.matchBitapScore(0, bestLoc, loc,
				pattern), scoreThreshold)
		}
	}

	// Initialise the bit arrays.
	matchmask := 1 << uint((len(pattern) - 1))
	bestLoc = -1

	var binMin, binMid int
	binMax := len(pattern) + len(text)
	lastRd := []int{}
	for d := 0; d < len(pattern); d++ {
		// Scan for the best match; each iteration allows for one more error. Run a binary search to determine how far from 'loc' we can stray at this error level.
		binMin = 0
		binMid = binMax
		for binMin < binMid {
			if dmp.matchBitapScore(d, loc+binMid, loc, pattern) <= scoreThreshold {
				binMin = binMid
			} else {
				binMax = binMid
			}
			binMid = (binMax-binMin)/2 + binMin
		}
		// Use the result from this iteration as the maximum for the next.
		binMax = binMid
		start := int(math.Max(1, float64(loc-binMid+1)))
		finish := int(math.Min(float64(loc+binMid), float64(len(text))) + float64(len(pattern)))

		rd := make([]int, finish+2)
		rd[finish+1] = (1 << uint(d)) - 1

		for j := finish; j >= start; j-- {
			var charMatch int
			if len(text) <= j-1 {
				// Out of range.
				charMatch = 0
			} else if _, ok := s[text[j-1]]; !ok {
				charMatch = 0
			} else {
				charMatch = s[text[j-1]]
			}

			if d == 0 {
				// First pass: exact match.
				rd[j] = ((rd[j+1] << 1) | 1) & charMatch
			} else {
				// Subsequent passes: fuzzy match.
				rd[j] = ((rd[j+1]<<1)|1)&charMatch | (((lastRd[j+1] | lastRd[j]) << 1) | 1) | lastRd[j+1]
			}
			if (rd[j] & matchmask) != 0 {
				score := dmp.matchBitapScore(d, j-1, loc, pattern)
				// This match will almost certainly be better than any existing match.  But check anyway.
				if score <= scoreThreshold {
					// Told you so.
					scoreThreshold = score
					bestLoc = j - 1
					if bestLoc > loc {
						// When passing loc, don't exceed our current distance from loc.
						start = int(math.Max(1, float64(2*loc-bestLoc)))
					} else {
						// Already passed loc, downhill from here on in.
						break
					}
				}
			}
		}
		if dmp.matchBitapScore(d+1, loc, loc, pattern) > scoreThreshold {
			// No hope for a (better) match at greater error levels.
			break
		}
		lastRd = rd
	}
	return bestLoc
}

// matchBitapScore computes and returns the score for a match with e errors and x location.
func (dmp *DiffMatchPatch) matchBitapScore(e, x, loc int, pattern string) float64 {
	accuracy := float64(e) / float64(len(pattern))
	proximity := math.Abs(float64(loc - x))
	if dmp.MatchDistance == 0 {
		// Dodge divide by zero error.
		if proximity == 0 {
			return accuracy
		}

		return 1.0
	}
	return accuracy + (proximity / float64(dmp.MatchDistance))
}

// MatchAlphabet initialises the alphabet for the Bitap algorithm.
func (dmp *DiffMatchPatch) MatchAlphabet(pattern string) map[byte]int {
	s := map[byte]int{}
	charPattern := []byte(pattern)
	for _, c := range charPattern {
		_, ok := s[c]
		if !ok {
			s[c] = 0
		}
	}
	i := 0

	for _, c := range charPattern {
		value := s[c] | int(uint(1)<<uint((len(pattern)-i-1)))
		s[c] = value
		i++
	}
	return s
}
