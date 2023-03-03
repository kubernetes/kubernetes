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
	"unicode"
)

var nameReplaceTable = map[rune]string{
	'@': "At ",
	'&': "And ",
	'|': "Pipe ",
	'$': "Dollar ",
	'!': "Bang ",
	'-': "",
	'_': "",
}

type (
	splitter struct {
		postSplitInitialismCheck bool
		initialisms              []string
	}

	splitterOption func(*splitter) *splitter
)

// split calls the splitter; splitter provides more control and post options
func split(str string) []string {
	lexems := newSplitter().split(str)
	result := make([]string, 0, len(lexems))

	for _, lexem := range lexems {
		result = append(result, lexem.GetOriginal())
	}

	return result

}

func (s *splitter) split(str string) []nameLexem {
	return s.toNameLexems(str)
}

func newSplitter(options ...splitterOption) *splitter {
	splitter := &splitter{
		postSplitInitialismCheck: false,
		initialisms:              initialisms,
	}

	for _, option := range options {
		splitter = option(splitter)
	}

	return splitter
}

// withPostSplitInitialismCheck allows to catch initialisms after main split process
func withPostSplitInitialismCheck(s *splitter) *splitter {
	s.postSplitInitialismCheck = true
	return s
}

type (
	initialismMatch struct {
		start, end int
		body       []rune
		complete   bool
	}
	initialismMatches []*initialismMatch
)

func (s *splitter) toNameLexems(name string) []nameLexem {
	nameRunes := []rune(name)
	matches := s.gatherInitialismMatches(nameRunes)
	return s.mapMatchesToNameLexems(nameRunes, matches)
}

func (s *splitter) gatherInitialismMatches(nameRunes []rune) initialismMatches {
	matches := make(initialismMatches, 0)

	for currentRunePosition, currentRune := range nameRunes {
		newMatches := make(initialismMatches, 0, len(matches))

		// check current initialism matches
		for _, match := range matches {
			if keepCompleteMatch := match.complete; keepCompleteMatch {
				newMatches = append(newMatches, match)
				continue
			}

			// drop failed match
			currentMatchRune := match.body[currentRunePosition-match.start]
			if !s.initialismRuneEqual(currentMatchRune, currentRune) {
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

			newMatches = append(newMatches, match)
		}

		// check for new initialism matches
		for _, initialism := range s.initialisms {
			initialismRunes := []rune(initialism)
			if s.initialismRuneEqual(initialismRunes[0], currentRune) {
				newMatches = append(newMatches, &initialismMatch{
					start:    currentRunePosition,
					body:     initialismRunes,
					complete: false,
				})
			}
		}

		matches = newMatches
	}

	return matches
}

func (s *splitter) mapMatchesToNameLexems(nameRunes []rune, matches initialismMatches) []nameLexem {
	nameLexems := make([]nameLexem, 0)

	var lastAcceptedMatch *initialismMatch
	for _, match := range matches {
		if !match.complete {
			continue
		}

		if firstMatch := lastAcceptedMatch == nil; firstMatch {
			nameLexems = append(nameLexems, s.breakCasualString(nameRunes[:match.start])...)
			nameLexems = append(nameLexems, s.breakInitialism(string(match.body)))

			lastAcceptedMatch = match

			continue
		}

		if overlappedMatch := match.start <= lastAcceptedMatch.end; overlappedMatch {
			continue
		}

		middle := nameRunes[lastAcceptedMatch.end+1 : match.start]
		nameLexems = append(nameLexems, s.breakCasualString(middle)...)
		nameLexems = append(nameLexems, s.breakInitialism(string(match.body)))

		lastAcceptedMatch = match
	}

	// we have not found any accepted matches
	if lastAcceptedMatch == nil {
		return s.breakCasualString(nameRunes)
	}

	if lastAcceptedMatch.end+1 != len(nameRunes) {
		rest := nameRunes[lastAcceptedMatch.end+1:]
		nameLexems = append(nameLexems, s.breakCasualString(rest)...)
	}

	return nameLexems
}

func (s *splitter) initialismRuneEqual(a, b rune) bool {
	return a == b
}

func (s *splitter) breakInitialism(original string) nameLexem {
	return newInitialismNameLexem(original, original)
}

func (s *splitter) breakCasualString(str []rune) []nameLexem {
	segments := make([]nameLexem, 0)
	currentSegment := ""

	addCasualNameLexem := func(original string) {
		segments = append(segments, newCasualNameLexem(original))
	}

	addInitialismNameLexem := func(original, match string) {
		segments = append(segments, newInitialismNameLexem(original, match))
	}

	addNameLexem := func(original string) {
		if s.postSplitInitialismCheck {
			for _, initialism := range s.initialisms {
				if upper(initialism) == upper(original) {
					addInitialismNameLexem(original, initialism)
					return
				}
			}
		}

		addCasualNameLexem(original)
	}

	for _, rn := range string(str) {
		if replace, found := nameReplaceTable[rn]; found {
			if currentSegment != "" {
				addNameLexem(currentSegment)
				currentSegment = ""
			}

			if replace != "" {
				addNameLexem(replace)
			}

			continue
		}

		if !unicode.In(rn, unicode.L, unicode.M, unicode.N, unicode.Pc) {
			if currentSegment != "" {
				addNameLexem(currentSegment)
				currentSegment = ""
			}

			continue
		}

		if unicode.IsUpper(rn) {
			if currentSegment != "" {
				addNameLexem(currentSegment)
			}
			currentSegment = ""
		}

		currentSegment += string(rn)
	}

	if currentSegment != "" {
		addNameLexem(currentSegment)
	}

	return segments
}
