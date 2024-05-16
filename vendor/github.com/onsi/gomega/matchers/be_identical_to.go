// untested sections: 2

package matchers

import (
	"fmt"
	"runtime"

	"github.com/onsi/gomega/format"
)

type BeIdenticalToMatcher struct {
	Expected interface{}
}

func (matcher *BeIdenticalToMatcher) Match(actual interface{}) (success bool, matchErr error) {
	if actual == nil && matcher.Expected == nil {
		return false, fmt.Errorf("Refusing to compare <nil> to <nil>.\nBe explicit and use BeNil() instead.  This is to avoid mistakes where both sides of an assertion are erroneously uninitialized.")
	}

	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				success = false
				matchErr = nil
			}
		}
	}()

	return actual == matcher.Expected, nil
}

func (matcher *BeIdenticalToMatcher) FailureMessage(actual interface{}) string {
	return format.Message(actual, "to be identical to", matcher.Expected)
}

func (matcher *BeIdenticalToMatcher) NegatedFailureMessage(actual interface{}) string {
	return format.Message(actual, "not to be identical to", matcher.Expected)
}
