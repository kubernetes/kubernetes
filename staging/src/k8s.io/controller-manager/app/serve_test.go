package app

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	componentbaseconfig "k8s.io/component-base/config"
)

func TestProfilingRoutesRegistered(t *testing.T) {
	conf := &componentbaseconfig.DebuggingConfiguration{
		EnableProfiling:           true,
		EnableContentionProfiling: false,
	}

	handler := NewBaseHandler(conf, http.NewServeMux())

	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL + "/debug/pprof/")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("expected /debug/pprof/ to be accessible, got err=%v, status=%d", err, resp.StatusCode)
	}
}

func TestContentionProfiling_EnablesMutexProfiling(t *testing.T) {
	conf := &componentbaseconfig.DebuggingConfiguration{
		EnableProfiling:           true,
		EnableContentionProfiling: true,
	}

	handler := NewBaseHandler(conf, http.NewServeMux())

	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL + "/debug/pprof/mutex?debug=1")
	if err != nil {
		t.Fatalf("failed to GET /debug/pprof/mutex: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(body, []byte("sampling period=1")) {
		t.Errorf("expected sampling period=1 in /debug/pprof/mutex, but got:\n%s", body)
	}
}
