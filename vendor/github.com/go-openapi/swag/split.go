// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"bytes"
	"sync"
	"unicode"
	"unicode/utf8"
)

type (
	splitter struct {
		initialisms              []string
		initialismsRunes         [][]rune
		initialismsUpperCased    [][]rune // initialisms cached in their trimmed, upper-cased version
		postSplitInitialismCheck bool
	}

	splitterOption func(*splitter)

	initialismMatch struct {
		body       []rune
		start, end int
		complete   bool
	}
	initialismMatches []initialismMatch
)

type (
	// memory pools of temporary objects.
	//
	// These are used to recycle temporarily allocated objects
	// and relieve the GC from undue pressure.

	matchesPool struct {
		*sync.Pool
	}

	buffersPool struct {
		*sync.Pool
	}

	lexemsPool struct {
		*sync.Pool
	}

	splittersPool struct {
		*sync.Pool
	}
)

var (
	// poolOfMatches holds temporary slices for recycling during the initialism match process
	poolOfMatches = matchesPool{
		Pool: &sync.Pool{
			New: func() any {
				s := make(initialismMatches, 0, maxAllocMatches)

				return &s
			},
		},
	}

	poolOfBuffers = buffersPool{
		Pool: &sync.Pool{
			New: func() any {
				return new(bytes.Buffer)
			},
		},
	}

	poolOfLexems = lexemsPool{
		Pool: &sync.Pool{
			New: func() any {
				s := make([]nameLexem, 0, maxAllocMatches)

				return &s
			},
		},
	}

	poolOfSplitters = splittersPool{
		Pool: &sync.Pool{
			New: func() any {
				s := newSplitter()

				return &s
			},
		},
	}
)

// nameReplaceTable finds a word representation for special characters.
func nameReplaceTable(r rune) (string, bool) {
	switch r {
	case '@':
		return "At ", true
	case '&':
		return "And ", true
	case '|':
		return "Pipe ", true
	case '$':
		return "Dollar ", true
	case '!':
		return "Bang ", true
	case '-':
		return "", true
	case '_':
		return "", true
	default:
		return "", false
	}
}

// split calls the splitter.
//
// Use newSplitter for more control and options
func split(str string) []string {
	s := poolOfSplitters.BorrowSplitter()
	lexems := s.split(str)
	result := make([]string, 0, len(*lexems))

	for _, lexem := range *lexems {
		result = append(result, lexem.GetOriginal())
	}
	poolOfLexems.RedeemLexems(lexems)
	poolOfSplitters.RedeemSplitter(s)

	return result

}

func newSplitter(options ...splitterOption) splitter {
	s := splitter{
		postSplitInitialismCheck: false,
		initialisms:              initialisms,
		initialismsRunes:         initialismsRunes,
		initialismsUpperCased:    initialismsUpperCased,
	}

	for _, option := range options {
		option(&s)
	}

	return s
}

// withPostSplitInitialismCheck allows to catch initialisms after main split process
func withPostSplitInitialismCheck(s *splitter) {
	s.postSplitInitialismCheck = true
}

func (p matchesPool) BorrowMatches() *initialismMatches {
	s := p.Get().(*initialismMatches)
	*s = (*s)[:0] // reset slice, keep allocated capacity

	return s
}

func (p buffersPool) BorrowBuffer(size int) *bytes.Buffer {
	s := p.Get().(*bytes.Buffer)
	s.Reset()

	if s.Cap() < size {
		s.Grow(size)
	}

	return s
}

func (p lexemsPool) BorrowLexems() *[]nameLexem {
	s := p.Get().(*[]nameLexem)
	*s = (*s)[:0] // reset slice, keep allocated capacity

	return s
}

func (p splittersPool) BorrowSplitter(options ...splitterOption) *splitter {
	s := p.Get().(*splitter)
	s.postSplitInitialismCheck = false // reset options
	for _, apply := range options {
		apply(s)
	}

	return s
}

func (p matchesPool) RedeemMatches(s *initialismMatches) {
	p.Put(s)
}

func (p buffersPool) RedeemBuffer(s *bytes.Buffer) {
	p.Put(s)
}

func (p lexemsPool) RedeemLexems(s *[]nameLexem) {
	p.Put(s)
}

func (p splittersPool) RedeemSplitter(s *splitter) {
	p.Put(s)
}

func (m initialismMatch) isZero() bool {
	return m.start == 0 && m.end == 0
}

func (s splitter) split(name string) *[]nameLexem {
	nameRunes := []rune(name)
	matches := s.gatherInitialismMatches(nameRunes)
	if matches == nil {
		return poolOfLexems.BorrowLexems()
	}

	return s.mapMatchesToNameLexems(nameRunes, matches)
}

func (s splitter) gatherInitialismMatches(nameRunes []rune) *initialismMatches {
	var matches *initialismMatches

	for currentRunePosition, currentRune := range nameRunes {
		// recycle these allocations as we loop over runes
		// with such recycling, only 2 slices should be allocated per call
		// instead of o(n).
		newMatches := poolOfMatches.BorrowMatches()

		// check current initialism matches
		if matches != nil { // skip first iteration
			for _, match := range *matches {
				if keepCompleteMatch := match.complete; keepCompleteMatch {
					*newMatches = append(*newMatches, match)
					continue
				}

				// drop failed match
				currentMatchRune := match.body[currentRunePosition-match.start]
				if currentMatchRune != currentRune {
					continue
				}

				// try to complete ongoing match
				if currentRunePosition-match.start == len(match.body)-1 {
					// we are close; the next step is to check the symbol ahead
					// if it is a small letter, then it is not the end of match
					// but beginning of the next word

					if currentRunePosition < len(nameRunes)-1 {
						nextRune := nameRunes[currentRunePosition+1]
						if newWord := unicode.IsLower(nextRune); newWord {
							// oh ok, it was the start of a new word
							continue
						}
					}

					match.complete = true
					match.end = currentRunePosition
				}

				*newMatches = append(*newMatches, match)
			}
		}

		// check for new initialism matches
		for i := range s.initialisms {
			initialismRunes := s.initialismsRunes[i]
			if initialismRunes[0] == currentRune {
				*newMatches = append(*newMatches, initialismMatch{
					start:    currentRunePosition,
					body:     initialismRunes,
					complete: false,
				})
			}
		}

		if matches != nil {
			poolOfMatches.RedeemMatches(matches)
		}
		matches = newMatches
	}

	// up to the caller to redeem this last slice
	return matches
}

func (s splitter) mapMatchesToNameLexems(nameRunes []rune, matches *initialismMatches) *[]nameLexem {
	nameLexems := poolOfLexems.BorrowLexems()

	var lastAcceptedMatch initialismMatch
	for _, match := range *matches {
		if !match.complete {
			continue
		}

		if firstMatch := lastAcceptedMatch.isZero(); firstMatch {
			s.appendBrokenDownCasualString(nameLexems, nameRunes[:match.start])
			*nameLexems = append(*nameLexems, s.breakInitialism(string(match.body)))

			lastAcceptedMatch = match

			continue
		}

		if overlappedMatch := match.start <= lastAcceptedMatch.end; overlappedMatch {
			continue
		}

		middle := nameRunes[lastAcceptedMatch.end+1 : match.start]
		s.appendBrokenDownCasualString(nameLexems, middle)
		*nameLexems = append(*nameLexems, s.breakInitialism(string(match.body)))

		lastAcceptedMatch = match
	}

	// we have not found any accepted matches
	if lastAcceptedMatch.isZero() {
		*nameLexems = (*nameLexems)[:0]
		s.appendBrokenDownCasualString(nameLexems, nameRunes)
	} else if lastAcceptedMatch.end+1 != len(nameRunes) {
		rest := nameRunes[lastAcceptedMatch.end+1:]
		s.appendBrokenDownCasualString(nameLexems, rest)
	}

	poolOfMatches.RedeemMatches(matches)

	return nameLexems
}

func (s splitter) breakInitialism(original string) nameLexem {
	return newInitialismNameLexem(original, original)
}

func (s splitter) appendBrokenDownCasualString(segments *[]nameLexem, str []rune) {
	currentSegment := poolOfBuffers.BorrowBuffer(len(str)) // unlike strings.Builder, bytes.Buffer initial storage can reused
	defer func() {
		poolOfBuffers.RedeemBuffer(currentSegment)
	}()

	addCasualNameLexem := func(original string) {
		*segments = append(*segments, newCasualNameLexem(original))
	}

	addInitialismNameLexem := func(original, match string) {
		*segments = append(*segments, newInitialismNameLexem(original, match))
	}

	var addNameLexem func(string)
	if s.postSplitInitialismCheck {
		addNameLexem = func(original string) {
			for i := range s.initialisms {
				if isEqualFoldIgnoreSpace(s.initialismsUpperCased[i], original) {
					addInitialismNameLexem(original, s.initialisms[i])

					return
				}
			}

			addCasualNameLexem(original)
		}
	} else {
		addNameLexem = addCasualNameLexem
	}

	for _, rn := range str {
		if replace, found := nameReplaceTable(rn); found {
			if currentSegment.Len() > 0 {
				addNameLexem(currentSegment.String())
				currentSegment.Reset()
			}

			if replace != "" {
				addNameLexem(replace)
			}

			continue
		}

		if !unicode.In(rn, unicode.L, unicode.M, unicode.N, unicode.Pc) {
			if currentSegment.Len() > 0 {
				addNameLexem(currentSegment.String())
				currentSegment.Reset()
			}

			continue
		}

		if unicode.IsUpper(rn) {
			if currentSegment.Len() > 0 {
				addNameLexem(currentSegment.String())
			}
			currentSegment.Reset()
		}

		currentSegment.WriteRune(rn)
	}

	if currentSegment.Len() > 0 {
		addNameLexem(currentSegment.String())
	}
}

// isEqualFoldIgnoreSpace is the same as strings.EqualFold, but
// it ignores leading and trailing blank spaces in the compared
// string.
//
// base is assumed to be composed of upper-cased runes, and be already
// trimmed.
//
// This code is heavily inspired from strings.EqualFold.
func isEqualFoldIgnoreSpace(base []rune, str string) bool {
	var i, baseIndex int
	// equivalent to b := []byte(str), but without data copy
	b := hackStringBytes(str)

	for i < len(b) {
		if c := b[i]; c < utf8.RuneSelf {
			// fast path for ASCII
			if c != ' ' && c != '\t' {
				break
			}
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if !unicode.IsSpace(r) {
			break
		}
		i += size
	}

	if i >= len(b) {
		return len(base) == 0
	}

	for _, baseRune := range base {
		if i >= len(b) {
			break
		}

		if c := b[i]; c < utf8.RuneSelf {
			// single byte rune case (ASCII)
			if baseRune >= utf8.RuneSelf {
				return false
			}

			baseChar := byte(baseRune)
			if c != baseChar &&
				!('a' <= c && c <= 'z' && c-'a'+'A' == baseChar) {
				return false
			}

			baseIndex++
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if unicode.ToUpper(r) != baseRune {
			return false
		}
		baseIndex++
		i += size
	}

	if baseIndex != len(base) {
		return false
	}

	// all passed: now we should only have blanks
	for i < len(b) {
		if c := b[i]; c < utf8.RuneSelf {
			// fast path for ASCII
			if c != ' ' && c != '\t' {
				return false
			}
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if !unicode.IsSpace(r) {
			return false
		}

		i += size
	}

	return true
}
