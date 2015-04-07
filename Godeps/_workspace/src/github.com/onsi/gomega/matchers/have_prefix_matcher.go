package matchers

import (
	"fmt"
	"github.com/onsi/gomega/format"
)

type HavePrefixMatcher struct {
	Prefix string
	Args   []interface{}
}

func (matcher *HavePrefixMatcher) Match(actual interface{}) (success bool, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return false, fmt.Errorf("HavePrefix matcher requires a string or stringer.  Got:\n%s", format.Object(actual, 1))
	}
	prefix := matcher.prefix()
	return len(actualString) >= len(prefix) && actualString[0:len(prefix)] == prefix, nil
}

func (matcher *HavePrefixMatcher) prefix() string {
	if len(matcher.Args) > 0 {
		return fmt.Sprintf(matcher.Prefix, matcher.Args...)
	}
	return matcher.Prefix
}

func (matcher *HavePrefixMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to have prefix", matcher.prefix())
}

func (matcher *HavePrefixMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to have prefix", matcher.prefix())
}
