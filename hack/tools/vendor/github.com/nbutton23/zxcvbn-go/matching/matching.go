package matching

import (
	"sort"

	"github.com/nbutton23/zxcvbn-go/adjacency"
	"github.com/nbutton23/zxcvbn-go/frequency"
	"github.com/nbutton23/zxcvbn-go/match"
)

var (
	dictionaryMatchers []match.Matcher
	matchers           []match.Matcher
	adjacencyGraphs    []adjacency.Graph
	l33tTable          adjacency.Graph

	sequences map[string]string
)

func init() {
	loadFrequencyList()
}

// Omnimatch runs all matchers against the password
func Omnimatch(password string, userInputs []string, filters ...func(match.Matcher) bool) (matches []match.Match) {

	//Can I run into the issue where nil is not equal to nil?
	if dictionaryMatchers == nil || adjacencyGraphs == nil {
		loadFrequencyList()
	}

	if userInputs != nil {
		userInputMatcher := buildDictMatcher("user_inputs", buildRankedDict(userInputs))
		matches = userInputMatcher(password)
	}

	for _, matcher := range matchers {
		shouldBeFiltered := false
		for i := range filters {
			if filters[i](matcher) {
				shouldBeFiltered = true
				break
			}
		}
		if !shouldBeFiltered {
			matches = append(matches, matcher.MatchingFunc(password)...)
		}
	}
	sort.Sort(match.Matches(matches))
	return matches
}

func loadFrequencyList() {

	for n, list := range frequency.Lists {
		dictionaryMatchers = append(dictionaryMatchers, match.Matcher{MatchingFunc: buildDictMatcher(n, buildRankedDict(list.List)), ID: n})
	}

	l33tTable = adjacency.GraphMap["l33t"]

	adjacencyGraphs = append(adjacencyGraphs, adjacency.GraphMap["qwerty"])
	adjacencyGraphs = append(adjacencyGraphs, adjacency.GraphMap["dvorak"])
	adjacencyGraphs = append(adjacencyGraphs, adjacency.GraphMap["keypad"])
	adjacencyGraphs = append(adjacencyGraphs, adjacency.GraphMap["macKeypad"])

	//l33tFilePath, _ := filepath.Abs("adjacency/L33t.json")
	//L33T_TABLE = adjacency.GetAdjancencyGraphFromFile(l33tFilePath, "l33t")

	sequences = make(map[string]string)
	sequences["lower"] = "abcdefghijklmnopqrstuvwxyz"
	sequences["upper"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	sequences["digits"] = "0123456789"

	matchers = append(matchers, dictionaryMatchers...)
	matchers = append(matchers, match.Matcher{MatchingFunc: spatialMatch, ID: spatialMatcherName})
	matchers = append(matchers, match.Matcher{MatchingFunc: repeatMatch, ID: repeatMatcherName})
	matchers = append(matchers, match.Matcher{MatchingFunc: sequenceMatch, ID: sequenceMatcherName})
	matchers = append(matchers, match.Matcher{MatchingFunc: l33tMatch, ID: L33TMatcherName})
	matchers = append(matchers, match.Matcher{MatchingFunc: dateSepMatcher, ID: dateSepMatcherName})
	matchers = append(matchers, match.Matcher{MatchingFunc: dateWithoutSepMatch, ID: dateWithOutSepMatcherName})

}
