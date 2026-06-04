// untested sections: 2

package matchers

import "github.com/onsi/gomega/format"

type BeNilMatcher struct {
}

func (matcher *BeNilMatcher) Match(actual any) (success bool, err error) {
	return isNil(actual), nil
}

func (matcher *BeNilMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be nil")
}

func (matcher *BeNilMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be nil")
}
