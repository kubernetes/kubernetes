package crl

import (
	"bytes"
	"encoding/json"
	"github.com/cloudflare/cfssl/api"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

const (
	cert       = "../../crl/testdata/caTwo.pem"
	key        = "../../crl/testdata/ca-keyTwo.pem"
	serialList = "../../crl/testdata/serialList"
	expiryTime = "2000"
)

type testJSON struct {
	Certificate        string
	SerialNumber       []string
	PrivateKey         string
	ExpiryTime         string
	ExpectedHTTPStatus int
	ExpectedSuccess    bool
}

var tester = testJSON{
	Certificate:        cert,
	SerialNumber:       []string{"1", "2", "3"},
	PrivateKey:         key,
	ExpiryTime:         "2000",
	ExpectedHTTPStatus: 200,
	ExpectedSuccess:    true,
}

func newTestHandler(t *testing.T) http.Handler {
	return NewHandler()
}

func TestNewHandler(t *testing.T) {
	newTestHandler(t)
}

func newCRLServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestHandler(t))
	return ts
}

func testCRLCreation(t *testing.T, issuingKey, certFile string, expiry string, serialList []string) (resp *http.Response, body []byte) {

	ts := newCRLServer(t)
	defer ts.Close()

	obj := map[string]interface{}{}

	if certFile != "" {
		c, err := ioutil.ReadFile(certFile)
		if err != nil {
			t.Fatal(err)
		}
		obj["certificate"] = string(c)
	}

	obj["serialNumber"] = serialList

	if issuingKey != "" {
		c, err := ioutil.ReadFile(issuingKey)
		if err != nil {
			t.Fatal(err)
		}
		obj["issuingKey"] = string(c)
	}

	obj["expireTime"] = expiry

	blob, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}

	resp, err = http.Post(ts.URL, "application/json", bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestCRL(t *testing.T) {
	resp, body := testCRLCreation(t, tester.PrivateKey, tester.Certificate, tester.ExpiryTime, tester.SerialNumber)
	if resp.StatusCode != tester.ExpectedHTTPStatus {
		t.Logf("expected: %d, have %d", tester.ExpectedHTTPStatus, resp.StatusCode)
		t.Fatal(resp.Status, tester.ExpectedHTTPStatus, string(body))
	}

	message := new(api.Response)
	err := json.Unmarshal(body, message)
	if err != nil {
		t.Logf("failed to read response body: %v", err)
		t.Fatal(resp.Status, tester.ExpectedHTTPStatus, message)
	}

}
