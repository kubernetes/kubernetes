package fakematcher

import "fmt"

type FakeMatcher struct {
	ReceivedActual  interface{}
	MatchesToReturn bool
	ErrToReturn     error
}

func (matcher *FakeMatcher) Match(actual interface{}) (bool, error) {
	matcher.ReceivedActual = actual

	return matcher.MatchesToReturn, matcher.ErrToReturn
}

func (matcher *FakeMatcher) FailureMessage(actual interface{}) string {
	return fmt.Sprintf("positive: %v", actual)
}

func (matcher *FakeMatcher) NegatedFailureMessage(actual interface{}) string {
	return fmt.Sprintf("negative: %v", actual)
}
