package matchers

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega/format"
	"gopkg.in/yaml.v3"
)

type MatchYAMLMatcher struct {
	YAMLToMatch      any
	firstFailurePath []any
}

func (matcher *MatchYAMLMatcher) Match(actual any) (success bool, err error) {
	actualString, expectedString, err := matcher.toStrings(actual)
	if err != nil {
		return false, err
	}

	var aval any
	var eval any

	if err := yaml.Unmarshal([]byte(actualString), &aval); err != nil {
		return false, fmt.Errorf("Actual '%s' should be valid YAML, but it is not.\nUnderlying error:%s", actualString, err)
	}
	if err := yaml.Unmarshal([]byte(expectedString), &eval); err != nil {
		return false, fmt.Errorf("Expected '%s' should be valid YAML, but it is not.\nUnderlying error:%s", expectedString, err)
	}

	var equal bool
	equal, matcher.firstFailurePath = deepEqual(aval, eval)
	return equal, nil
}

func (matcher *MatchYAMLMatcher) FailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.toNormalisedStrings(actual)
	return formattedMessage(format.Message(actualString, "to match YAML of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchYAMLMatcher) NegatedFailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.toNormalisedStrings(actual)
	return formattedMessage(format.Message(actualString, "not to match YAML of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchYAMLMatcher) toNormalisedStrings(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, expectedString, err := matcher.toStrings(actual)
	return normalise(actualString), normalise(expectedString), err
}

func normalise(input string) string {
	var val any
	err := yaml.Unmarshal([]byte(input), &val)
	if err != nil {
		panic(err) // unreachable since Match already calls Unmarshal
	}
	output, err := yaml.Marshal(val)
	if err != nil {
		panic(err) // untested section, unreachable since we Unmarshal above
	}
	return strings.TrimSpace(string(output))
}

func (matcher *MatchYAMLMatcher) toStrings(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return "", "", fmt.Errorf("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got actual:\n%s", format.Object(actual, 1))
	}
	expectedString, ok := toString(matcher.YAMLToMatch)
	if !ok {
		return "", "", fmt.Errorf("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got expected:\n%s", format.Object(matcher.YAMLToMatch, 1))
	}

	return actualString, expectedString, nil
}
