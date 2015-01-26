package matchers

import (
	"fmt"
	"github.com/onsi/gomega/format"
	"strings"
)

type ContainSubstringMatcher struct {
	Substr string
	Args   []interface{}
}

func (matcher *ContainSubstringMatcher) Match(actual interface{}) (success bool, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return false, fmt.Errorf("ContainSubstring matcher requires a string or stringer.  Got:\n%s", format.Object(actual, 1))
	}

	return strings.Contains(actualString, matcher.stringToMatch()), nil
}

func (matcher *ContainSubstringMatcher) stringToMatch() string {
	stringToMatch := matcher.Substr
	if len(matcher.Args) > 0 {
		stringToMatch = fmt.Sprintf(matcher.Substr, matcher.Args...)
	}
	return stringToMatch
}

func (matcher *ContainSubstringMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to contain substring", matcher.stringToMatch())
}

func (matcher *ContainSubstringMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to contain substring", matcher.stringToMatch())
}
