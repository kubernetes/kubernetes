package health

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coreos/pkg/httputil"
)

type boolChecker bool

func (b boolChecker) Healthy() error {
	if b {
		return nil
	}
	return errors.New("Unhealthy")
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func TestCheck(t *testing.T) {
	for i, test := range []struct {
		checks   []Checkable
		expected string
	}{
		{[]Checkable{}, ""},

		{[]Checkable{boolChecker(true)}, ""},

		{[]Checkable{boolChecker(true), boolChecker(true)}, ""},

		{[]Checkable{boolChecker(true), boolChecker(false)}, "Unhealthy"},

		{[]Checkable{boolChecker(true), boolChecker(false), boolChecker(false)}, "multiple health check failure: [Unhealthy Unhealthy]"},
	} {
		err := Check(test.checks)

		if errString(err) != test.expected {
			t.Errorf("case %d: want %v, got %v", i, test.expected, errString(err))
		}
	}
}

func TestHandlerFunc(t *testing.T) {
	for i, test := range []struct {
		checker         Checker
		method          string
		expectedStatus  string
		expectedCode    int
		expectedMessage string
	}{
		{
			Checker{
				Checks: []Checkable{
					boolChecker(true),
				},
			},
			"GET",
			"ok",
			http.StatusOK,
			"",
		},

		// Wrong method.
		{
			Checker{
				Checks: []Checkable{
					boolChecker(true),
				},
			},
			"POST",
			"",
			http.StatusMethodNotAllowed,
			"GET only acceptable method",
		},

		// Health check fails.
		{
			Checker{
				Checks: []Checkable{
					boolChecker(false),
				},
			},
			"GET",
			"error",
			http.StatusInternalServerError,
			"Unhealthy",
		},

		// Health check fails, with overridden ErrorHandler.
		{
			Checker{
				Checks: []Checkable{
					boolChecker(false),
				},
				UnhealthyHandler: func(w http.ResponseWriter, r *http.Request, err error) {
					httputil.WriteJSONResponse(w,
						http.StatusInternalServerError, StatusResponse{
							Status: "error",
							Details: &StatusResponseDetails{
								Code:    http.StatusInternalServerError,
								Message: "Override!",
							},
						})
				},
			},
			"GET",
			"error",
			http.StatusInternalServerError,
			"Override!",
		},

		// Health check succeeds, with overridden SuccessHandler.
		{
			Checker{
				Checks: []Checkable{
					boolChecker(true),
				},
				HealthyHandler: func(w http.ResponseWriter, r *http.Request) {
					httputil.WriteJSONResponse(w,
						http.StatusOK, StatusResponse{
							Status: "okey-dokey",
						})
				},
			},
			"GET",
			"okey-dokey",
			http.StatusOK,
			"",
		},
	} {
		w := httptest.NewRecorder()
		r := &http.Request{}
		r.Method = test.method
		test.checker.ServeHTTP(w, r)
		if w.Code != test.expectedCode {
			t.Errorf("case %d: w.code == %v, want %v", i, w.Code, test.expectedCode)
		}

		if test.expectedStatus == "" {
			// This is to handle the wrong-method case, when the
			// body of the response is empty.
			continue
		}

		statusMap := make(map[string]interface{})
		err := json.Unmarshal(w.Body.Bytes(), &statusMap)
		if err != nil {
			t.Fatalf("case %d: failed to Unmarshal response body: %v", i, err)
		}

		status, ok := statusMap["status"].(string)
		if !ok {
			t.Errorf("case %d: status not present or not a string in json: %q", i, w.Body.Bytes())
		}
		if status != test.expectedStatus {
			t.Errorf("case %d: status == %v, want %v", i, status, test.expectedStatus)
		}

		detailMap, ok := statusMap["details"].(map[string]interface{})
		if test.expectedMessage != "" {
			if !ok {
				t.Fatalf("case %d: could not find/unmarshal detailMap", i)
			}
			message, ok := detailMap["message"].(string)
			if !ok {
				t.Fatalf("case %d: message not present or not a string in json: %q",
					i, w.Body.Bytes())
			}
			if message != test.expectedMessage {
				t.Errorf("case %d: message == %v, want %v", i, message, test.expectedMessage)
			}

			code, ok := detailMap["code"].(float64)
			if !ok {
				t.Fatalf("case %d: code not present or not an int in json: %q",
					i, w.Body.Bytes())
			}
			if int(code) != test.expectedCode {
				t.Errorf("case %d: code == %v, want %v", i, code, test.expectedCode)
			}

		} else {
			if ok {
				t.Errorf("case %d: unwanted detailMap present: %q", i, detailMap)
			}
		}

	}
}
