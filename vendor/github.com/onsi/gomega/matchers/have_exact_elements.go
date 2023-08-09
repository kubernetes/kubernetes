package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type mismatchFailure struct {
	failure string
	index   int
}

type HaveExactElementsMatcher struct {
	Elements         []interface{}
	mismatchFailures []mismatchFailure
	missingIndex     int
	extraIndex       int
}

func (matcher *HaveExactElementsMatcher) Match(actual interface{}) (success bool, err error) {
	matcher.resetState()

	if isMap(actual) {
		return false, fmt.Errorf("error")
	}

	matchers := matchers(matcher.Elements)
	values := valuesOf(actual)

	lenMatchers := len(matchers)
	lenValues := len(values)

	for i := 0; i < lenMatchers || i < lenValues; i++ {
		if i >= lenMatchers {
			matcher.extraIndex = i
			continue
		}

		if i >= lenValues {
			matcher.missingIndex = i
			return
		}

		elemMatcher := matchers[i].(omegaMatcher)
		match, err := elemMatcher.Match(values[i])
		if err != nil || !match {
			matcher.mismatchFailures = append(matcher.mismatchFailures, mismatchFailure{
				index:   i,
				failure: elemMatcher.FailureMessage(values[i]),
			})
		}
	}

	return matcher.missingIndex+matcher.extraIndex+len(matcher.mismatchFailures) == 0, nil
}

func (matcher *HaveExactElementsMatcher) FailureMessage(actual interface{}) (message string) {
	message = format.Message(actual, "to have exact elements with", presentable(matcher.Elements))
	if matcher.missingIndex > 0 {
		message = fmt.Sprintf("%s\nthe missing elements start from index %d", message, matcher.missingIndex)
	}
	if matcher.extraIndex > 0 {
		message = fmt.Sprintf("%s\nthe extra elements start from index %d", message, matcher.extraIndex)
	}
	if len(matcher.mismatchFailures) != 0 {
		message = fmt.Sprintf("%s\nthe mismatch indexes were:", message)
	}
	for _, mismatch := range matcher.mismatchFailures {
		message = fmt.Sprintf("%s\n%d: %s", message, mismatch.index, mismatch.failure)
	}
	return
}

func (matcher *HaveExactElementsMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to contain elements", presentable(matcher.Elements))
}

func (matcher *HaveExactElementsMatcher) resetState() {
	matcher.mismatchFailures = nil
	matcher.missingIndex = 0
	matcher.extraIndex = 0
}
