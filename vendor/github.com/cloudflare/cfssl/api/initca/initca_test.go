package initca

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cloudflare/cfssl/csr"
)

func csrData(t *testing.T) *bytes.Reader {
	req := &csr.CertificateRequest{
		Names: []csr.Name{
			{
				C:  "US",
				ST: "California",
				L:  "San Francisco",
				O:  "CloudFlare",
				OU: "Systems Engineering",
			},
		},
		CN:         "cloudflare.com",
		Hosts:      []string{"cloudflare.com"},
		KeyRequest: csr.NewBasicKeyRequest(),
	}
	csrBytes, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(csrBytes)
}

func TestInitCARESTfulVerbs(t *testing.T) {
	ts := httptest.NewServer(NewHandler())
	data := csrData(t)
	// POST should work.
	req, _ := http.NewRequest("POST", ts.URL, data)
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusOK {
		t.Fatal(resp.Status)
	}

	// Test GET, PUT, DELETE and whatever, expect 400 errors.
	req, _ = http.NewRequest("GET", ts.URL, data)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("PUT", ts.URL, data)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("DELETE", ts.URL, data)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
	req, _ = http.NewRequest("WHATEVER", ts.URL, data)
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatal(resp.Status)
	}
}

func TestBadRequestBody(t *testing.T) {
	ts := httptest.NewServer(NewHandler())
	req, _ := http.NewRequest("POST", ts.URL, nil)
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode == http.StatusOK {
		t.Fatal(resp.Status)
	}
}

func TestBadRequestBody_2(t *testing.T) {
	ts := httptest.NewServer(NewHandler())
	r := &csr.CertificateRequest{}
	csrBytes, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	data := bytes.NewReader(csrBytes)
	req, _ := http.NewRequest("POST", ts.URL, data)
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode == http.StatusOK {
		t.Fatal(resp.Status)
	}
}
