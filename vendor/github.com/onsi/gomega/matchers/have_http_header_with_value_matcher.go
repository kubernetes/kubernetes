package matchers

import (
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

type HaveHTTPHeaderWithValueMatcher struct {
	Header string
	Value  interface{}
}

func (matcher *HaveHTTPHeaderWithValueMatcher) Match(actual interface{}) (success bool, err error) {
	headerValue, err := matcher.extractHeader(actual)
	if err != nil {
		return false, err
	}

	headerMatcher, err := matcher.getSubMatcher()
	if err != nil {
		return false, err
	}

	return headerMatcher.Match(headerValue)
}

func (matcher *HaveHTTPHeaderWithValueMatcher) FailureMessage(actual interface{}) string {
	headerValue, err := matcher.extractHeader(actual)
	if err != nil {
		panic(err) // protected by Match()
	}

	headerMatcher, err := matcher.getSubMatcher()
	if err != nil {
		panic(err) // protected by Match()
	}

	diff := format.IndentString(headerMatcher.FailureMessage(headerValue), 1)
	return fmt.Sprintf("HTTP header %q:\n%s", matcher.Header, diff)
}

func (matcher *HaveHTTPHeaderWithValueMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	headerValue, err := matcher.extractHeader(actual)
	if err != nil {
		panic(err) // protected by Match()
	}

	headerMatcher, err := matcher.getSubMatcher()
	if err != nil {
		panic(err) // protected by Match()
	}

	diff := format.IndentString(headerMatcher.NegatedFailureMessage(headerValue), 1)
	return fmt.Sprintf("HTTP header %q:\n%s", matcher.Header, diff)
}

func (matcher *HaveHTTPHeaderWithValueMatcher) getSubMatcher() (types.GomegaMatcher, error) {
	switch m := matcher.Value.(type) {
	case string:
		return &EqualMatcher{Expected: matcher.Value}, nil
	case types.GomegaMatcher:
		return m, nil
	default:
		return nil, fmt.Errorf("HaveHTTPHeaderWithValue matcher must be passed a string or a GomegaMatcher. Got:\n%s", format.Object(matcher.Value, 1))
	}
}

func (matcher *HaveHTTPHeaderWithValueMatcher) extractHeader(actual interface{}) (string, error) {
	switch r := actual.(type) {
	case *http.Response:
		return r.Header.Get(matcher.Header), nil
	case *httptest.ResponseRecorder:
		return r.Result().Header.Get(matcher.Header), nil
	default:
		return "", fmt.Errorf("HaveHTTPHeaderWithValue matcher expects *http.Response or *httptest.ResponseRecorder. Got:\n%s", format.Object(actual, 1))
	}
}
