// untested sections: 2

package matchers

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega/format"
)

type ContainSubstringMatcher struct {
	Substr string
	Args   []any
}

func (matcher *ContainSubstringMatcher) Match(actual any) (success bool, err error) {
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

func (matcher *ContainSubstringMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to contain substring", matcher.stringToMatch())
}

func (matcher *ContainSubstringMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to contain substring", matcher.stringToMatch())
}
