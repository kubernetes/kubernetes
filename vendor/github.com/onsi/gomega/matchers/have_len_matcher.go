package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type HaveLenMatcher struct {
	Count int
}

func (matcher *HaveLenMatcher) Match(actual interface{}) (success bool, err error) {
	length, ok := lengthOf(actual)
	if !ok {
		return false, fmt.Errorf("HaveLen matcher expects a string/array/map/channel/slice.  Got:\n%s", format.Object(actual, 1))
	}

	return length == matcher.Count, nil
}

func (matcher *HaveLenMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected\n%s\nto have length %d", format.Object(actual, 1), matcher.Count)
}

func (matcher *HaveLenMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected\n%s\nnot to have length %d", format.Object(actual, 1), matcher.Count)
}
