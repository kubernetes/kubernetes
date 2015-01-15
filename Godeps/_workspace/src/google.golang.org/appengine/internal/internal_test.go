package internal

import (
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInstallingHealthChecker(t *testing.T) {
	try := func(desc string, mux *http.ServeMux, wantCode int, wantBody string) {
		installHealthChecker(mux)
		srv := httptest.NewServer(mux)
		defer srv.Close()

		resp, err := http.Get(srv.URL + "/_ah/health")
		if err != nil {
			t.Errorf("%s: http.Get: %v", desc, err)
			return
		}
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%s: reading body: %v", desc, err)
			return
		}

		if resp.StatusCode != wantCode {
			t.Errorf("%s: got HTTP %d, want %d", desc, resp.StatusCode, wantCode)
			return
		}
		if wantBody != "" && string(body) != wantBody {
			t.Errorf("%s: got HTTP body %q, want %q", desc, body, wantBody)
			return
		}
	}

	// If there's no handlers, or only a root handler, a health checker should be installed.
	try("empty mux", http.NewServeMux(), 200, "ok")
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "root handler")
	})
	try("mux with root handler", mux, 200, "ok")

	// If there's a custom health check handler, one should not be installed.
	mux = http.NewServeMux()
	mux.HandleFunc("/_ah/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(418)
		io.WriteString(w, "I'm short and stout!")
	})
	try("mux with custom health checker", mux, 418, "I'm short and stout!")
}
