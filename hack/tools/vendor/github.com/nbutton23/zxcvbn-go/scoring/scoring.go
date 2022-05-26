package scoring

import (
	"fmt"
	"github.com/nbutton23/zxcvbn-go/entropy"
	"github.com/nbutton23/zxcvbn-go/match"
	"github.com/nbutton23/zxcvbn-go/utils/math"
	"math"
	"sort"
)

const (
	//for a hash function like bcrypt/scrypt/PBKDF2, 10ms per guess is a safe lower bound.
	//(usually a guess would take longer -- this assumes fast hardware and a small work factor.)
	//adjust for your site accordingly if you use another hash function, possibly by
	//several orders of magnitude!
	singleGuess     float64 = 0.010
	numAttackers    float64 = 100 //Cores used to make guesses
	secondsPerGuess float64 = singleGuess / numAttackers
)

// MinEntropyMatch is the lowest entropy match found
type MinEntropyMatch struct {
	Password         string
	Entropy          float64
	MatchSequence    []match.Match
	CrackTime        float64
	CrackTimeDisplay string
	Score            int
	CalcTime         float64
}

/*
MinimumEntropyMatchSequence returns the minimum entropy

    Takes a list of overlapping matches, returns the non-overlapping sublist with
    minimum entropy. O(nm) dp alg for length-n password with m candidate matches.
*/
func MinimumEntropyMatchSequence(password string, matches []match.Match) MinEntropyMatch {
	bruteforceCardinality := float64(entropy.CalcBruteForceCardinality(password))
	upToK := make([]float64, len(password))
	backPointers := make([]match.Match, len(password))

	for k := 0; k < len(password); k++ {
		upToK[k] = get(upToK, k-1) + math.Log2(bruteforceCardinality)

		for _, match := range matches {
			if match.J != k {
				continue
			}

			i, j := match.I, match.J
			//see if best entropy up to i-1 + entropy of match is less that current min at j
			upTo := get(upToK, i-1)
			candidateEntropy := upTo + match.Entropy

			if candidateEntropy < upToK[j] {
				upToK[j] = candidateEntropy
				match.Entropy = candidateEntropy
				backPointers[j] = match
			}
		}
	}

	//walk backwards and decode the best sequence
	var matchSequence []match.Match
	passwordLen := len(password)
	passwordLen--
	for k := passwordLen; k >= 0; {
		match := backPointers[k]
		if match.Pattern != "" {
			matchSequence = append(matchSequence, match)
			k = match.I - 1

		} else {
			k--
		}

	}
	sort.Sort(match.Matches(matchSequence))

	makeBruteForceMatch := func(i, j int) match.Match {
		return match.Match{Pattern: "bruteforce",
			I:       i,
			J:       j,
			Token:   password[i : j+1],
			Entropy: math.Log2(math.Pow(bruteforceCardinality, float64(j-i)))}

	}

	k := 0
	var matchSequenceCopy []match.Match
	for _, match := range matchSequence {
		i, j := match.I, match.J
		if i-k > 0 {
			matchSequenceCopy = append(matchSequenceCopy, makeBruteForceMatch(k, i-1))
		}
		k = j + 1
		matchSequenceCopy = append(matchSequenceCopy, match)
	}

	if k < len(password) {
		matchSequenceCopy = append(matchSequenceCopy, makeBruteForceMatch(k, len(password)-1))
	}
	var minEntropy float64
	if len(password) == 0 {
		minEntropy = float64(0)
	} else {
		minEntropy = upToK[len(password)-1]
	}

	crackTime := roundToXDigits(entropyToCrackTime(minEntropy), 3)
	return MinEntropyMatch{Password: password,
		Entropy:          roundToXDigits(minEntropy, 3),
		MatchSequence:    matchSequenceCopy,
		CrackTime:        crackTime,
		CrackTimeDisplay: displayTime(crackTime),
		Score:            crackTimeToScore(crackTime)}

}
func get(a []float64, i int) float64 {
	if i < 0 || i >= len(a) {
		return float64(0)
	}

	return a[i]
}

func entropyToCrackTime(entropy float64) float64 {
	crackTime := (0.5 * math.Pow(float64(2), entropy)) * secondsPerGuess

	return crackTime
}

func roundToXDigits(number float64, digits int) float64 {
	return zxcvbnmath.Round(number, .5, digits)
}

func displayTime(seconds float64) string {
	formater := "%.1f %s"
	minute := float64(60)
	hour := minute * float64(60)
	day := hour * float64(24)
	month := day * float64(31)
	year := month * float64(12)
	century := year * float64(100)

	if seconds < minute {
		return "instant"
	} else if seconds < hour {
		return fmt.Sprintf(formater, (1 + math.Ceil(seconds/minute)), "minutes")
	} else if seconds < day {
		return fmt.Sprintf(formater, (1 + math.Ceil(seconds/hour)), "hours")
	} else if seconds < month {
		return fmt.Sprintf(formater, (1 + math.Ceil(seconds/day)), "days")
	} else if seconds < year {
		return fmt.Sprintf(formater, (1 + math.Ceil(seconds/month)), "months")
	} else if seconds < century {
		return fmt.Sprintf(formater, (1 + math.Ceil(seconds/century)), "years")
	} else {
		return "centuries"
	}
}

func crackTimeToScore(seconds float64) int {
	if seconds < math.Pow(10, 2) {
		return 0
	} else if seconds < math.Pow(10, 4) {
		return 1
	} else if seconds < math.Pow(10, 6) {
		return 2
	} else if seconds < math.Pow(10, 8) {
		return 3
	}

	return 4
}
