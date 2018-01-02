package generator

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/signer/local"
)

const (
	testCaFile    = "testdata/ca.pem"
	testCaKeyFile = "testdata/ca_key.pem"
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

func TestGeneratorRESTfulVerbs(t *testing.T) {
	handler, _ := NewHandler(CSRValidate)
	ts := httptest.NewServer(handler)
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

func TestCSRValidate(t *testing.T) {
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
		Hosts:      []string{},
		KeyRequest: csr.NewBasicKeyRequest(),
	}
	err := CSRValidate(req)
	if err != nil {
		t.Fatal("There should be not an error for missing Hosts parameter")
	}
}

func TestNewCertGeneratorHandlerFromSigner(t *testing.T) {
	var expiry = 1 * time.Minute
	var CAConfig = &config.Config{
		Signing: &config.Signing{
			Profiles: map[string]*config.SigningProfile{
				"signature": {
					Usage:  []string{"digital signature"},
					Expiry: expiry,
				},
			},
			Default: &config.SigningProfile{
				Usage:        []string{"cert sign", "crl sign"},
				ExpiryString: "43800h",
				Expiry:       expiry,
				CA:           true,

				ClientProvidesSerialNumbers: true,
			},
		},
	}
	s, err := local.NewSignerFromFile(testCaFile, testCaKeyFile, CAConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}

	h := NewCertGeneratorHandlerFromSigner(CSRValidate, s)
	_, ok := h.(http.Handler)
	if !ok {
		t.Fatal("A HTTP handler has not been returned")
	}
}
