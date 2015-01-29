package matchers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/onsi/gomega/format"
	"reflect"
)

type MatchJSONMatcher struct {
	JSONToMatch interface{}
}

func (matcher *MatchJSONMatcher) Match(actual interface{}) (success bool, err error) {
	actualString, expectedString, err := matcher.prettyPrint(actual)
	if err != nil {
		return false, err
	}

	var aval interface{}
	var eval interface{}

	// this is guarded by prettyPrint
	json.Unmarshal([]byte(actualString), &aval)
	json.Unmarshal([]byte(expectedString), &eval)

	return reflect.DeepEqual(aval, eval), nil
}

func (matcher *MatchJSONMatcher) FailureMessage(actual interface{}) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return format.Message(actualString, "to match JSON of", expectedString)
}

func (matcher *MatchJSONMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return format.Message(actualString, "not to match JSON of", expectedString)
}

func (matcher *MatchJSONMatcher) prettyPrint(actual interface{}) (actualFormatted, expectedFormatted string, err error) {
	actualString, aok := toString(actual)
	expectedString, eok := toString(matcher.JSONToMatch)

	if !(aok && eok) {
		return "", "", fmt.Errorf("MatchJSONMatcher matcher requires a string or stringer.  Got:\n%s", format.Object(actual, 1))
	}

	abuf := new(bytes.Buffer)
	ebuf := new(bytes.Buffer)

	if err := json.Indent(abuf, []byte(actualString), "", "  "); err != nil {
		return "", "", err
	}

	if err := json.Indent(ebuf, []byte(expectedString), "", "  "); err != nil {
		return "", "", err
	}

	return actualString, expectedString, nil
}
