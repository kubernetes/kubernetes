// untested sections: 2

package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type BeEmptyMatcher struct {
}

func (matcher *BeEmptyMatcher) Match(actual interface{}) (success bool, err error) {
	length, ok := lengthOf(actual)
	if !ok {
		return false, fmt.Errorf("BeEmpty matcher expects a string/array/map/channel/slice.  Got:\n%s", format.Object(actual, 1))
	}

	return length == 0, nil
}

func (matcher *BeEmptyMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to be empty")
}

func (matcher *BeEmptyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to be empty")
}
