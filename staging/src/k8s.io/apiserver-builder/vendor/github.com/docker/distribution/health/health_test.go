package health

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestReturns200IfThereAreNoChecks ensures that the result code of the health
// endpoint is 200 if there are not currently registered checks.
func TestReturns200IfThereAreNoChecks(t *testing.T) {
	recorder := httptest.NewRecorder()

	req, err := http.NewRequest("GET", "https://fakeurl.com/debug/health", nil)
	if err != nil {
		t.Errorf("Failed to create request.")
	}

	StatusHandler(recorder, req)

	if recorder.Code != 200 {
		t.Errorf("Did not get a 200.")
	}
}

// TestReturns500IfThereAreErrorChecks ensures that the result code of the
// health endpoint is 500 if there are health checks with errors
func TestReturns503IfThereAreErrorChecks(t *testing.T) {
	recorder := httptest.NewRecorder()

	req, err := http.NewRequest("GET", "https://fakeurl.com/debug/health", nil)
	if err != nil {
		t.Errorf("Failed to create request.")
	}

	// Create a manual error
	Register("some_check", CheckFunc(func() error {
		return errors.New("This Check did not succeed")
	}))

	StatusHandler(recorder, req)

	if recorder.Code != 503 {
		t.Errorf("Did not get a 503.")
	}
}

// TestHealthHandler ensures that our handler implementation correct protects
// the web application when things aren't so healthy.
func TestHealthHandler(t *testing.T) {
	// clear out existing checks.
	DefaultRegistry = NewRegistry()

	// protect an http server
	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	// wrap it in our health handler
	handler = Handler(handler)

	// use this swap check status
	updater := NewStatusUpdater()
	Register("test_check", updater)

	// now, create a test server
	server := httptest.NewServer(handler)

	checkUp := func(t *testing.T, message string) {
		resp, err := http.Get(server.URL)
		if err != nil {
			t.Fatalf("error getting success status: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusNoContent {
			t.Fatalf("unexpected response code from server when %s: %d != %d", message, resp.StatusCode, http.StatusNoContent)
		}
		// NOTE(stevvooe): we really don't care about the body -- the format is
		// not standardized or supported, yet.
	}

	checkDown := func(t *testing.T, message string) {
		resp, err := http.Get(server.URL)
		if err != nil {
			t.Fatalf("error getting down status: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("unexpected response code from server when %s: %d != %d", message, resp.StatusCode, http.StatusServiceUnavailable)
		}
	}

	// server should be up
	checkUp(t, "initial health check")

	// now, we fail the health check
	updater.Update(fmt.Errorf("the server is now out of commission"))
	checkDown(t, "server should be down") // should be down

	// bring server back up
	updater.Update(nil)
	checkUp(t, "when server is back up") // now we should be back up.
}
