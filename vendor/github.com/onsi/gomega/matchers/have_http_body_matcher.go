package matchers

import (
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/internal/gutil"
	"github.com/onsi/gomega/types"
)

type HaveHTTPBodyMatcher struct {
	Expected   interface{}
	cachedBody []byte
}

func (matcher *HaveHTTPBodyMatcher) Match(actual interface{}) (bool, error) {
	body, err := matcher.body(actual)
	if err != nil {
		return false, err
	}

	switch e := matcher.Expected.(type) {
	case string:
		return (&EqualMatcher{Expected: e}).Match(string(body))
	case []byte:
		return (&EqualMatcher{Expected: e}).Match(body)
	case types.GomegaMatcher:
		return e.Match(body)
	default:
		return false, fmt.Errorf("HaveHTTPBody matcher expects string, []byte, or GomegaMatcher. Got:\n%s", format.Object(matcher.Expected, 1))
	}
}

func (matcher *HaveHTTPBodyMatcher) FailureMessage(actual interface{}) (message string) {
	body, err := matcher.body(actual)
	if err != nil {
		return fmt.Sprintf("failed to read body: %s", err)
	}

	switch e := matcher.Expected.(type) {
	case string:
		return (&EqualMatcher{Expected: e}).FailureMessage(string(body))
	case []byte:
		return (&EqualMatcher{Expected: e}).FailureMessage(body)
	case types.GomegaMatcher:
		return e.FailureMessage(body)
	default:
		return fmt.Sprintf("HaveHTTPBody matcher expects string, []byte, or GomegaMatcher. Got:\n%s", format.Object(matcher.Expected, 1))
	}
}

func (matcher *HaveHTTPBodyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	body, err := matcher.body(actual)
	if err != nil {
		return fmt.Sprintf("failed to read body: %s", err)
	}

	switch e := matcher.Expected.(type) {
	case string:
		return (&EqualMatcher{Expected: e}).NegatedFailureMessage(string(body))
	case []byte:
		return (&EqualMatcher{Expected: e}).NegatedFailureMessage(body)
	case types.GomegaMatcher:
		return e.NegatedFailureMessage(body)
	default:
		return fmt.Sprintf("HaveHTTPBody matcher expects string, []byte, or GomegaMatcher. Got:\n%s", format.Object(matcher.Expected, 1))
	}
}

// body returns the body. It is cached because once we read it in Match()
// the Reader is closed and it is not readable again in FailureMessage()
// or NegatedFailureMessage()
func (matcher *HaveHTTPBodyMatcher) body(actual interface{}) ([]byte, error) {
	if matcher.cachedBody != nil {
		return matcher.cachedBody, nil
	}

	body := func(a *http.Response) ([]byte, error) {
		if a.Body != nil {
			defer a.Body.Close()
			var err error
			matcher.cachedBody, err = gutil.ReadAll(a.Body)
			if err != nil {
				return nil, fmt.Errorf("error reading response body: %w", err)
			}
		}
		return matcher.cachedBody, nil
	}

	switch a := actual.(type) {
	case *http.Response:
		return body(a)
	case *httptest.ResponseRecorder:
		return body(a.Result())
	default:
		return nil, fmt.Errorf("HaveHTTPBody matcher expects *http.Response or *httptest.ResponseRecorder. Got:\n%s", format.Object(actual, 1))
	}

}
