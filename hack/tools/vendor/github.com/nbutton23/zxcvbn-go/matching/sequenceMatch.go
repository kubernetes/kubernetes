package matching

import (
	"strings"

	"github.com/nbutton23/zxcvbn-go/entropy"
	"github.com/nbutton23/zxcvbn-go/match"
)

const sequenceMatcherName = "SEQ"

//FilterSequenceMatcher can be pass to zxcvbn-go.PasswordStrength to skip that matcher
func FilterSequenceMatcher(m match.Matcher) bool {
	return m.ID == sequenceMatcherName
}

func sequenceMatch(password string) []match.Match {
	var matches []match.Match
	for i := 0; i < len(password); {
		j := i + 1
		var seq string
		var seqName string
		seqDirection := 0
		for seqCandidateName, seqCandidate := range sequences {
			iN := strings.Index(seqCandidate, string(password[i]))
			var jN int
			if j < len(password) {
				jN = strings.Index(seqCandidate, string(password[j]))
			} else {
				jN = -1
			}

			if iN > -1 && jN > -1 {
				direction := jN - iN
				if direction == 1 || direction == -1 {
					seq = seqCandidate
					seqName = seqCandidateName
					seqDirection = direction
					break
				}
			}

		}

		if seq != "" {
			for {
				var prevN, curN int
				if j < len(password) {
					prevChar, curChar := password[j-1], password[j]
					prevN, curN = strings.Index(seq, string(prevChar)), strings.Index(seq, string(curChar))
				}

				if j == len(password) || curN-prevN != seqDirection {
					if j-i > 2 {
						matchSequence := match.Match{
							Pattern:        "sequence",
							I:              i,
							J:              j - 1,
							Token:          password[i:j],
							DictionaryName: seqName,
						}

						matchSequence.Entropy = entropy.SequenceEntropy(matchSequence, len(seq), (seqDirection == 1))
						matches = append(matches, matchSequence)
					}
					break
				} else {
					j++
				}

			}
		}
		i = j
	}
	return matches
}
