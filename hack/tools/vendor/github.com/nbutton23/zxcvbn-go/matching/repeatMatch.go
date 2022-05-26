package matching

import (
	"strings"

	"github.com/nbutton23/zxcvbn-go/entropy"
	"github.com/nbutton23/zxcvbn-go/match"
)

const repeatMatcherName = "REPEAT"

//FilterRepeatMatcher can be pass to zxcvbn-go.PasswordStrength to skip that matcher
func FilterRepeatMatcher(m match.Matcher) bool {
	return m.ID == repeatMatcherName
}

func repeatMatch(password string) []match.Match {
	var matches []match.Match

	//Loop through password. if current == prev currentStreak++ else if currentStreak > 2 {buildMatch; currentStreak = 1} prev = current
	var current, prev string
	currentStreak := 1
	var i int
	var char rune
	for i, char = range password {
		current = string(char)
		if i == 0 {
			prev = current
			continue
		}

		if strings.ToLower(current) == strings.ToLower(prev) {
			currentStreak++

		} else if currentStreak > 2 {
			iPos := i - currentStreak
			jPos := i - 1
			matchRepeat := match.Match{
				Pattern:        "repeat",
				I:              iPos,
				J:              jPos,
				Token:          password[iPos : jPos+1],
				DictionaryName: prev}
			matchRepeat.Entropy = entropy.RepeatEntropy(matchRepeat)
			matches = append(matches, matchRepeat)
			currentStreak = 1
		} else {
			currentStreak = 1
		}

		prev = current
	}

	if currentStreak > 2 {
		iPos := i - currentStreak + 1
		jPos := i
		matchRepeat := match.Match{
			Pattern:        "repeat",
			I:              iPos,
			J:              jPos,
			Token:          password[iPos : jPos+1],
			DictionaryName: prev}
		matchRepeat.Entropy = entropy.RepeatEntropy(matchRepeat)
		matches = append(matches, matchRepeat)
	}
	return matches
}
