package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

type OrMatcher struct {
	Matchers []types.GomegaMatcher

	// state
	firstSuccessfulMatcher types.GomegaMatcher
}

func (m *OrMatcher) Match(actual interface{}) (success bool, err error) {
	m.firstSuccessfulMatcher = nil
	for _, matcher := range m.Matchers {
		success, err := matcher.Match(actual)
		if err != nil {
			return false, err
		}
		if success {
			m.firstSuccessfulMatcher = matcher
			return true, nil
		}
	}
	return false, nil
}

func (m *OrMatcher) FailureMessage(actual interface{}) (message string) {
	// not the most beautiful list of matchers, but not bad either...
	return format.Message(actual, fmt.Sprintf("To satisfy at least one of these matchers: %s", m.Matchers))
}

func (m *OrMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return m.firstSuccessfulMatcher.NegatedFailureMessage(actual)
}

func (m *OrMatcher) MatchMayChangeInTheFuture(actual interface{}) bool {
	/*
		Example with 3 matchers: A, B, C

		Match evaluates them: F, T, <?>  => T
		So match is currently T, what should MatchMayChangeInTheFuture() return?
		Seems like it only depends on B, since currently B MUST change to allow the result to become F

		Match eval: F, F, F  => F
		So match is currently F, what should MatchMayChangeInTheFuture() return?
		Seems to depend on ANY of them being able to change to T.
	*/

	if m.firstSuccessfulMatcher != nil {
		// one of the matchers succeeded.. it must be able to change in order to affect the result
		return types.MatchMayChangeInTheFuture(m.firstSuccessfulMatcher, actual)
	} else {
		// so all matchers failed.. Any one of them changing would change the result.
		for _, matcher := range m.Matchers {
			if types.MatchMayChangeInTheFuture(matcher, actual) {
				return true
			}
		}
		return false // none of were going to change
	}
}
