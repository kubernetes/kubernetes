package filters

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithNotReady(t *testing.T) {
	const warning = `299 - "The apiserver was still initializing, while this request was being served"`

	tests := []struct {
		name               string
		requestURL         string
		hasBeenReady       bool
		user               *user.DefaultInfo
		handlerInvoked     int
		retryAfterExpected string
		warningExpected    string
		statusCodeexpected int
	}{
		{
			name:               "the apiserver is fully initialized",
			hasBeenReady:       true,
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
		},
		{
			name:               "the apiserver is initializing, local loopback",
			hasBeenReady:       false,
			user:               &user.DefaultInfo{Name: user.APIServerUser},
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, exempt debugger group",
			hasBeenReady:       false,
			user:               &user.DefaultInfo{Groups: []string{"system:authenticated", notReadyDebuggerGroup}},
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, readyz",
			requestURL:         "/readyz?verbose=1",
			user:               &user.DefaultInfo{},
			hasBeenReady:       false,
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, healthz",
			requestURL:         "/healthz?verbose=1",
			user:               &user.DefaultInfo{},
			hasBeenReady:       false,
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, livez",
			requestURL:         "/livez?verbose=1",
			user:               &user.DefaultInfo{},
			hasBeenReady:       false,
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, metrics",
			requestURL:         "/metrics",
			user:               &user.DefaultInfo{},
			hasBeenReady:       false,
			handlerInvoked:     1,
			statusCodeexpected: http.StatusOK,
			warningExpected:    warning,
		},
		{
			name:               "the apiserver is initializing, non-exempt request",
			hasBeenReady:       false,
			user:               &user.DefaultInfo{Groups: []string{"system:authenticated", "system:masters"}},
			statusCodeexpected: http.StatusServiceUnavailable,
			retryAfterExpected: "5",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			hasBeenReadyCh := make(chan struct{})
			if test.hasBeenReady {
				close(hasBeenReadyCh)
			} else {
				defer close(hasBeenReadyCh)
			}

			var handlerInvoked int
			handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				handlerInvoked++
				w.WriteHeader(http.StatusOK)
			})

			if len(test.requestURL) == 0 {
				test.requestURL = "/api/v1/namespaces"
			}
			req, err := http.NewRequest(http.MethodGet, test.requestURL, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if test.user != nil {
				req = req.WithContext(request.WithUser(req.Context(), test.user))
			}
			w := httptest.NewRecorder()

			withNotReady := WithNotReady(handler, hasBeenReadyCh)
			withNotReady = genericapifilters.WithWarningRecorder(withNotReady)
			withNotReady.ServeHTTP(w, req)

			if test.handlerInvoked != handlerInvoked {
				t.Errorf("expected the handler to be invoked: %d times, but got: %d", test.handlerInvoked, handlerInvoked)
			}
			if test.statusCodeexpected != w.Code {
				t.Errorf("expected Response Status Code: %d, but got: %d", test.statusCodeexpected, w.Code)
			}

			retryAfterGot := w.Header().Get("Retry-After")
			if test.retryAfterExpected != retryAfterGot {
				t.Errorf("expected Retry-After: %q, but got: %q", test.retryAfterExpected, retryAfterGot)
			}

			warningGot := w.Header().Get("Warning")
			if test.warningExpected != warningGot {
				t.Errorf("expected Warning: %s, but got: %s", test.warningExpected, warningGot)
			}

		})
	}
}
