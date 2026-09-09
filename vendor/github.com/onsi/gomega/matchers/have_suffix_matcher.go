package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type HaveSuffixMatcher struct {
	Suffix string
	Args   []any
}

func (matcher *HaveSuffixMatcher) Match(actual any) (success bool, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return false, fmt.Errorf("HaveSuffix matcher requires a string or stringer.  Got:\n%s", format.Object(actual, 1))
	}
	suffix := matcher.suffix()
	return len(actualString) >= len(suffix) && actualString[len(actualString)-len(suffix):] == suffix, nil
}

func (matcher *HaveSuffixMatcher) suffix() string {
	if len(matcher.Args) > 0 {
		return fmt.Sprintf(matcher.Suffix, matcher.Args...)
	}
	return matcher.Suffix
}

func (matcher *HaveSuffixMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to have suffix", matcher.suffix())
}

func (matcher *HaveSuffixMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to have suffix", matcher.suffix())
}
