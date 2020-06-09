// untested sections: 2

package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type HaveCapMatcher struct {
	Count int
}

func (matcher *HaveCapMatcher) Match(actual interface{}) (success bool, err error) {
	length, ok := capOf(actual)
	if !ok {
		return false, fmt.Errorf("HaveCap matcher expects a array/channel/slice.  Got:\n%s", format.Object(actual, 1))
	}

	return length == matcher.Count, nil
}

func (matcher *HaveCapMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected\n%s\nto have capacity %d", format.Object(actual, 1), matcher.Count)
}

func (matcher *HaveCapMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected\n%s\nnot to have capacity %d", format.Object(actual, 1), matcher.Count)
}
