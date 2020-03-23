package matchers

import (
	"fmt"
	"regexp"

	"github.com/onsi/gomega/format"
)

type MatchRegexpMatcher struct {
	Regexp string
	Args   []interface{}
}

func (matcher *MatchRegexpMatcher) Match(actual interface{}) (success bool, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return false, fmt.Errorf("RegExp matcher requires a string or stringer.\nGot:%s", format.Object(actual, 1))
	}

	match, err := regexp.Match(matcher.regexp(), []byte(actualString))
	if err != nil {
		return false, fmt.Errorf("RegExp match failed to compile with error:\n\t%s", err.Error())
	}

	return match, nil
}

func (matcher *MatchRegexpMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to match regular expression", matcher.regexp())
}

func (matcher *MatchRegexpMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match regular expression", matcher.regexp())
}

func (matcher *MatchRegexpMatcher) regexp() string {
	re := matcher.Regexp
	if len(matcher.Args) > 0 {
		re = fmt.Sprintf(matcher.Regexp, matcher.Args...)
	}
	return re
}
