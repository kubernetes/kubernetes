package matching

import (
	"strings"

	"github.com/nbutton23/zxcvbn-go/adjacency"
	"github.com/nbutton23/zxcvbn-go/entropy"
	"github.com/nbutton23/zxcvbn-go/match"
)

const spatialMatcherName = "SPATIAL"

//FilterSpatialMatcher can be pass to zxcvbn-go.PasswordStrength to skip that matcher
func FilterSpatialMatcher(m match.Matcher) bool {
	return m.ID == spatialMatcherName
}

func spatialMatch(password string) (matches []match.Match) {
	for _, graph := range adjacencyGraphs {
		if graph.Graph != nil {
			matches = append(matches, spatialMatchHelper(password, graph)...)
		}
	}
	return matches
}

func spatialMatchHelper(password string, graph adjacency.Graph) (matches []match.Match) {

	for i := 0; i < len(password)-1; {
		j := i + 1
		lastDirection := -99 //an int that it should never be!
		turns := 0
		shiftedCount := 0

		for {
			prevChar := password[j-1]
			found := false
			foundDirection := -1
			curDirection := -1
			//My graphs seem to be wrong. . . and where the hell is qwerty
			adjacents := graph.Graph[string(prevChar)]
			//Consider growing pattern by one character if j hasn't gone over the edge
			if j < len(password) {
				curChar := password[j]
				for _, adj := range adjacents {
					curDirection++

					if strings.Index(adj, string(curChar)) != -1 {
						found = true
						foundDirection = curDirection

						if strings.Index(adj, string(curChar)) == 1 {
							//index 1 in the adjacency means the key is shifted, 0 means unshifted: A vs a, % vs 5, etc.
							//for example, 'q' is adjacent to the entry '2@'. @ is shifted w/ index 1, 2 is unshifted.
							shiftedCount++
						}

						if lastDirection != foundDirection {
							//adding a turn is correct even in the initial case when last_direction is null:
							//every spatial pattern starts with a turn.
							turns++
							lastDirection = foundDirection
						}
						break
					}
				}
			}

			//if the current pattern continued, extend j and try to grow again
			if found {
				j++
			} else {
				//otherwise push the pattern discovered so far, if any...
				//don't consider length 1 or 2 chains.
				if j-i > 2 {
					matchSpc := match.Match{Pattern: "spatial", I: i, J: j - 1, Token: password[i:j], DictionaryName: graph.Name}
					matchSpc.Entropy = entropy.SpatialEntropy(matchSpc, turns, shiftedCount)
					matches = append(matches, matchSpc)
				}
				//. . . and then start a new search from the rest of the password
				i = j
				break
			}
		}

	}
	return matches
}
