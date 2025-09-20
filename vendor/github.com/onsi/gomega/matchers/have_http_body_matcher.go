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
	Expected       any
	cachedResponse any
	cachedBody     []byte
}

func (matcher *HaveHTTPBodyMatcher) Match(actual any) (bool, error) {
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

func (matcher *HaveHTTPBodyMatcher) FailureMessage(actual any) (message string) {
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

func (matcher *HaveHTTPBodyMatcher) NegatedFailureMessage(actual any) (message string) {
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
func (matcher *HaveHTTPBodyMatcher) body(actual any) ([]byte, error) {
	if matcher.cachedResponse == actual && matcher.cachedBody != nil {
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
		matcher.cachedResponse = a
		return body(a)
	case *httptest.ResponseRecorder:
		matcher.cachedResponse = a
		return body(a.Result())
	default:
		return nil, fmt.Errorf("HaveHTTPBody matcher expects *http.Response or *httptest.ResponseRecorder. Got:\n%s", format.Object(actual, 1))
	}

}
