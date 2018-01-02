package scan

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

var (
	handler, _ = NewHandler("")
	ts         = httptest.NewServer(handler)
)

func TestBadRequest(t *testing.T) {
	// Test request with no host
	req, _ := http.NewRequest("GET", ts.URL, nil)
	resp, _ := http.DefaultClient.Do(req)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatal(resp.Status)
	}
}

func TestScanRESTfulVerbs(t *testing.T) {
	// GET should work
	req, _ := http.NewRequest("GET", ts.URL, nil)
	data := req.URL.Query()
	data.Add("host", "cloudflare.com")
	req.URL.RawQuery = data.Encode()
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusOK {
		t.Fatal(resp.Status)
	}

	// POST, PUT, DELETE, WHATEVER should return 400 errors
	req, _ = http.NewRequest("POST", ts.URL, nil)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("DELETE", ts.URL, nil)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("PUT", ts.URL, nil)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("WHATEVER", ts.URL, nil)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
}

func TestNewInfoHandler(t *testing.T) {
	handler := NewInfoHandler()
	if handler == nil {
		t.Fatal("Handler error")
	}
}
