package matchers

import (
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/onsi/gomega/format"
)

type HaveHTTPStatusMatcher struct {
	Expected interface{}
}

func (matcher *HaveHTTPStatusMatcher) Match(actual interface{}) (success bool, err error) {
	var resp *http.Response
	switch a := actual.(type) {
	case *http.Response:
		resp = a
	case *httptest.ResponseRecorder:
		resp = a.Result()
	default:
		return false, fmt.Errorf("HaveHTTPStatus matcher expects *http.Response or *httptest.ResponseRecorder. Got:\n%s", format.Object(actual, 1))
	}

	switch e := matcher.Expected.(type) {
	case int:
		return resp.StatusCode == e, nil
	case string:
		return resp.Status == e, nil
	}

	return false, fmt.Errorf("HaveHTTPStatus matcher must be passed an int or a string. Got:\n%s", format.Object(matcher.Expected, 1))
}

func (matcher *HaveHTTPStatusMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to have HTTP status", matcher.Expected)
}

func (matcher *HaveHTTPStatusMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to have HTTP status", matcher.Expected)
}
