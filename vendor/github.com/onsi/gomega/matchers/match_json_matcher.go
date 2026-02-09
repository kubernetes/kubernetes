package matchers

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/onsi/gomega/format"
)

type MatchJSONMatcher struct {
	JSONToMatch      any
	firstFailurePath []any
}

func (matcher *MatchJSONMatcher) Match(actual any) (success bool, err error) {
	actualString, expectedString, err := matcher.prettyPrint(actual)
	if err != nil {
		return false, err
	}

	var aval any
	var eval any

	// this is guarded by prettyPrint
	json.Unmarshal([]byte(actualString), &aval)
	json.Unmarshal([]byte(expectedString), &eval)
	var equal bool
	equal, matcher.firstFailurePath = deepEqual(aval, eval)
	return equal, nil
}

func (matcher *MatchJSONMatcher) FailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return formattedMessage(format.Message(actualString, "to match JSON of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchJSONMatcher) NegatedFailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return formattedMessage(format.Message(actualString, "not to match JSON of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchJSONMatcher) prettyPrint(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return "", "", fmt.Errorf("MatchJSONMatcher matcher requires a string, stringer, or []byte.  Got actual:\n%s", format.Object(actual, 1))
	}
	expectedString, ok := toString(matcher.JSONToMatch)
	if !ok {
		return "", "", fmt.Errorf("MatchJSONMatcher matcher requires a string, stringer, or []byte.  Got expected:\n%s", format.Object(matcher.JSONToMatch, 1))
	}

	abuf := new(bytes.Buffer)
	ebuf := new(bytes.Buffer)

	if err := json.Indent(abuf, []byte(actualString), "", "  "); err != nil {
		return "", "", fmt.Errorf("Actual '%s' should be valid JSON, but it is not.\nUnderlying error:%s", actualString, err)
	}

	if err := json.Indent(ebuf, []byte(expectedString), "", "  "); err != nil {
		return "", "", fmt.Errorf("Expected '%s' should be valid JSON, but it is not.\nUnderlying error:%s", expectedString, err)
	}

	return abuf.String(), ebuf.String(), nil
}
