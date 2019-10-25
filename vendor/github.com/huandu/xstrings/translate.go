// Copyright 2015 Huan Du. All rights reserved.
// Licensed under the MIT license that can be found in the LICENSE file.

package xstrings

import (
	"bytes"
	"unicode"
	"unicode/utf8"
)

type runeRangeMap struct {
	FromLo rune // Lower bound of range map.
	FromHi rune // An inclusive higher bound of range map.
	ToLo   rune
	ToHi   rune
}

type runeDict struct {
	Dict [unicode.MaxASCII + 1]rune
}

type runeMap map[rune]rune

// Translator can translate string with pre-compiled from and to patterns.
// If a from/to pattern pair needs to be used more than once, it's recommended
// to create a Translator and reuse it.
type Translator struct {
	quickDict  *runeDict       // A quick dictionary to look up rune by index. Only availabe for latin runes.
	runeMap    runeMap         // Rune map for translation.
	ranges     []*runeRangeMap // Ranges of runes.
	mappedRune rune            // If mappedRune >= 0, all matched runes are translated to the mappedRune.
	reverted   bool            // If to pattern is empty, all matched characters will be deleted.
	hasPattern bool
}

// NewTranslator creates new Translator through a from/to pattern pair.
func NewTranslator(from, to string) *Translator {
	tr := &Translator{}

	if from == "" {
		return tr
	}

	reverted := from[0] == '^'
	deletion := len(to) == 0

	if reverted {
		from = from[1:]
	}

	var fromStart, fromEnd, fromRangeStep rune
	var toStart, toEnd, toRangeStep rune
	var fromRangeSize, toRangeSize rune
	var singleRunes []rune

	// Update the to rune range.
	updateRange := func() {
		// No more rune to read in the to rune pattern.
		if toEnd == utf8.RuneError {
			return
		}

		if toRangeStep == 0 {
			to, toStart, toEnd, toRangeStep = nextRuneRange(to, toEnd)
			return
		}

		// Current range is not empty. Consume 1 rune from start.
		if toStart != toEnd {
			toStart += toRangeStep
			return
		}

		// No more rune. Repeat the last rune.
		if to == "" {
			toEnd = utf8.RuneError
			return
		}

		// Both start and end are used. Read two more runes from the to pattern.
		to, toStart, toEnd, toRangeStep = nextRuneRange(to, utf8.RuneError)
	}

	if deletion {
		toStart = utf8.RuneError
		toEnd = utf8.RuneError
	} else {
		// If from pattern is reverted, only the last rune in the to pattern will be used.
		if reverted {
			var size int

			for len(to) > 0 {
				toStart, size = utf8.DecodeRuneInString(to)
				to = to[size:]
			}

			toEnd = utf8.RuneError
		} else {
			to, toStart, toEnd, toRangeStep = nextRuneRange(to, utf8.RuneError)
		}
	}

	fromEnd = utf8.RuneError

	for len(from) > 0 {
		from, fromStart, fromEnd, fromRangeStep = nextRuneRange(from, fromEnd)

		// fromStart is a single character. Just map it with a rune in the to pattern.
		if fromRangeStep == 0 {
			singleRunes = tr.addRune(fromStart, toStart, singleRunes)
			updateRange()
			continue
		}

		for toEnd != utf8.RuneError && fromStart != fromEnd {
			// If mapped rune is a single character instead of a range, simply shift first
			// rune in the range.
			if toRangeStep == 0 {
				singleRunes = tr.addRune(fromStart, toStart, singleRunes)
				updateRange()
				fromStart += fromRangeStep
				continue
			}

			fromRangeSize = (fromEnd - fromStart) * fromRangeStep
			toRangeSize = (toEnd - toStart) * toRangeStep

			// Not enough runes in the to pattern. Need to read more.
			if fromRangeSize > toRangeSize {
				fromStart, toStart = tr.addRuneRange(fromStart, fromStart+toRangeSize*fromRangeStep, toStart, toEnd, singleRunes)
				fromStart += fromRangeStep
				updateRange()

				// Edge case: If fromRangeSize == toRangeSize + 1, the last fromStart value needs be considered
				// as a single rune.
				if fromStart == fromEnd {
					singleRunes = tr.addRune(fromStart, toStart, singleRunes)
					updateRange()
				}

				continue
			}

			fromStart, toStart = tr.addRuneRange(fromStart, fromEnd, toStart, toStart+fromRangeSize*toRangeStep, singleRunes)
			updateRange()
			break
		}

		if fromStart == fromEnd {
			fromEnd = utf8.RuneError
			continue
		}

		fromStart, toStart = tr.addRuneRange(fromStart, fromEnd, toStart, toStart, singleRunes)
		fromEnd = utf8.RuneError
	}

	if fromEnd != utf8.RuneError {
		singleRunes = tr.addRune(fromEnd, toStart, singleRunes)
	}

	tr.reverted = reverted
	tr.mappedRune = -1
	tr.hasPattern = true

	// Translate RuneError only if in deletion or reverted mode.
	if deletion || reverted {
		tr.mappedRune = toStart
	}

	return tr
}

func (tr *Translator) addRune(from, to rune, singleRunes []rune) []rune {
	if from <= unicode.MaxASCII {
		if tr.quickDict == nil {
			tr.quickDict = &runeDict{}
		}

		tr.quickDict.Dict[from] = to
	} else {
		if tr.runeMap == nil {
			tr.runeMap = make(runeMap)
		}

		tr.runeMap[from] = to
	}

	singleRunes = append(singleRunes, from)
	return singleRunes
}

func (tr *Translator) addRuneRange(fromLo, fromHi, toLo, toHi rune, singleRunes []rune) (rune, rune) {
	var r rune
	var rrm *runeRangeMap

	if fromLo < fromHi {
		rrm = &runeRangeMap{
			FromLo: fromLo,
			FromHi: fromHi,
			ToLo:   toLo,
			ToHi:   toHi,
		}
	} else {
		rrm = &runeRangeMap{
			FromLo: fromHi,
			FromHi: fromLo,
			ToLo:   toHi,
			ToHi:   toLo,
		}
	}

	// If there is any single rune conflicts with this rune range, clear single rune record.
	for _, r = range singleRunes {
		if rrm.FromLo <= r && r <= rrm.FromHi {
			if r <= unicode.MaxASCII {
				tr.quickDict.Dict[r] = 0
			} else {
				delete(tr.runeMap, r)
			}
		}
	}

	tr.ranges = append(tr.ranges, rrm)
	return fromHi, toHi
}

func nextRuneRange(str string, last rune) (remaining string, start, end rune, rangeStep rune) {
	var r rune
	var size int

	remaining = str
	escaping := false
	isRange := false

	for len(remaining) > 0 {
		r, size = utf8.DecodeRuneInString(remaining)
		remaining = remaining[size:]

		// Parse special characters.
		if !escaping {
			if r == '\\' {
				escaping = true
				continue
			}

			if r == '-' {
				// Ignore slash at beginning of string.
				if last == utf8.RuneError {
					continue
				}

				start = last
				isRange = true
				continue
			}
		}

		escaping = false

		if last != utf8.RuneError {
			// This is a range which start and end are the same.
			// Considier it as a normal character.
			if isRange && last == r {
				isRange = false
				continue
			}

			start = last
			end = r

			if isRange {
				if start < end {
					rangeStep = 1
				} else {
					rangeStep = -1
				}
			}

			return
		}

		last = r
	}

	start = last
	end = utf8.RuneError
	return
}

// Translate str with a from/to pattern pair.
//
// See comment in Translate function for usage and samples.
func (tr *Translator) Translate(str string) string {
	if !tr.hasPattern || str == "" {
		return str
	}

	var r rune
	var size int
	var needTr bool

	orig := str

	var output *bytes.Buffer

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)
		r, needTr = tr.TranslateRune(r)

		if needTr && output == nil {
			output = allocBuffer(orig, str)
		}

		if r != utf8.RuneError && output != nil {
			output.WriteRune(r)
		}

		str = str[size:]
	}

	// No character is translated.
	if output == nil {
		return orig
	}

	return output.String()
}

// TranslateRune return translated rune and true if r matches the from pattern.
// If r doesn't match the pattern, original r is returned and translated is false.
func (tr *Translator) TranslateRune(r rune) (result rune, translated bool) {
	switch {
	case tr.quickDict != nil:
		if r <= unicode.MaxASCII {
			result = tr.quickDict.Dict[r]

			if result != 0 {
				translated = true

				if tr.mappedRune >= 0 {
					result = tr.mappedRune
				}

				break
			}
		}

		fallthrough

	case tr.runeMap != nil:
		var ok bool

		if result, ok = tr.runeMap[r]; ok {
			translated = true

			if tr.mappedRune >= 0 {
				result = tr.mappedRune
			}

			break
		}

		fallthrough

	default:
		var rrm *runeRangeMap
		ranges := tr.ranges

		for i := len(ranges) - 1; i >= 0; i-- {
			rrm = ranges[i]

			if rrm.FromLo <= r && r <= rrm.FromHi {
				translated = true

				if tr.mappedRune >= 0 {
					result = tr.mappedRune
					break
				}

				if rrm.ToLo < rrm.ToHi {
					result = rrm.ToLo + r - rrm.FromLo
				} else if rrm.ToLo > rrm.ToHi {
					// ToHi can be smaller than ToLo if range is from higher to lower.
					result = rrm.ToLo - r + rrm.FromLo
				} else {
					result = rrm.ToLo
				}

				break
			}
		}
	}

	if tr.reverted {
		if !translated {
			result = tr.mappedRune
		}

		translated = !translated
	}

	if !translated {
		result = r
	}

	return
}

// HasPattern returns true if Translator has one pattern at least.
func (tr *Translator) HasPattern() bool {
	return tr.hasPattern
}

// Translate str with the characters defined in from replaced by characters defined in to.
//
// From and to are patterns representing a set of characters. Pattern is defined as following.
//
//     * Special characters
//       * '-' means a range of runes, e.g.
//         * "a-z" means all characters from 'a' to 'z' inclusive;
//         * "z-a" means all characters from 'z' to 'a' inclusive.
//       * '^' as first character means a set of all runes excepted listed, e.g.
//         * "^a-z" means all characters except 'a' to 'z' inclusive.
//       * '\' escapes special characters.
//     * Normal character represents itself, e.g. "abc" is a set including 'a', 'b' and 'c'.
//
// Translate will try to find a 1:1 mapping from from to to.
// If to is smaller than from, last rune in to will be used to map "out of range" characters in from.
//
// Note that '^' only works in the from pattern. It will be considered as a normal character in the to pattern.
//
// If the to pattern is an empty string, Translate works exactly the same as Delete.
//
// Samples:
//     Translate("hello", "aeiou", "12345")    => "h2ll4"
//     Translate("hello", "a-z", "A-Z")        => "HELLO"
//     Translate("hello", "z-a", "a-z")        => "svool"
//     Translate("hello", "aeiou", "*")        => "h*ll*"
//     Translate("hello", "^l", "*")           => "**ll*"
//     Translate("hello ^ world", `\^lo`, "*") => "he*** * w*r*d"
func Translate(str, from, to string) string {
	tr := NewTranslator(from, to)
	return tr.Translate(str)
}

// Delete runes in str matching the pattern.
// Pattern is defined in Translate function.
//
// Samples:
//     Delete("hello", "aeiou") => "hll"
//     Delete("hello", "a-k")   => "llo"
//     Delete("hello", "^a-k")  => "he"
func Delete(str, pattern string) string {
	tr := NewTranslator(pattern, "")
	return tr.Translate(str)
}

// Count how many runes in str match the pattern.
// Pattern is defined in Translate function.
//
// Samples:
//     Count("hello", "aeiou") => 3
//     Count("hello", "a-k")   => 3
//     Count("hello", "^a-k")  => 2
func Count(str, pattern string) int {
	if pattern == "" || str == "" {
		return 0
	}

	var r rune
	var size int
	var matched bool

	tr := NewTranslator(pattern, "")
	cnt := 0

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)
		str = str[size:]

		if _, matched = tr.TranslateRune(r); matched {
			cnt++
		}
	}

	return cnt
}

// Squeeze deletes adjacent repeated runes in str.
// If pattern is not empty, only runes matching the pattern will be squeezed.
//
// Samples:
//     Squeeze("hello", "")             => "helo"
//     Squeeze("hello", "m-z")          => "hello"
//     Squeeze("hello   world", " ")    => "hello world"
func Squeeze(str, pattern string) string {
	var last, r rune
	var size int
	var skipSqueeze, matched bool
	var tr *Translator
	var output *bytes.Buffer

	orig := str
	last = -1

	if len(pattern) > 0 {
		tr = NewTranslator(pattern, "")
	}

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)

		// Need to squeeze the str.
		if last == r && !skipSqueeze {
			if tr != nil {
				if _, matched = tr.TranslateRune(r); !matched {
					skipSqueeze = true
				}
			}

			if output == nil {
				output = allocBuffer(orig, str)
			}

			if skipSqueeze {
				output.WriteRune(r)
			}
		} else {
			if output != nil {
				output.WriteRune(r)
			}

			last = r
			skipSqueeze = false
		}

		str = str[size:]
	}

	if output == nil {
		return orig
	}

	return output.String()
}
