package bundle

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cloudflare/cfssl/api"
)

const (
	testCaBundleFile     = "../testdata/ca-bundle.pem"
	testIntBundleFile    = "../testdata/int-bundle.pem"
	testLeafCertFile     = "../testdata/leaf.pem"
	testLeafKeyFile      = "../testdata/leaf.key"
	testLeafWrongKeyFile = "../testdata/leaf.badkey"
	testBrokenCertFile   = "../testdata/broken.pem"
)

func newTestHandler(t *testing.T) (h http.Handler) {
	h, err := NewHandler(testCaBundleFile, testIntBundleFile)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newBundleServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestHandler(t))
	return ts
}

func testBundleFile(t *testing.T, domain, ip, certFile, keyFile, flavor string) (resp *http.Response, body []byte) {
	ts := newBundleServer(t)
	defer ts.Close()
	var certPEM, keyPEM []byte
	if certFile != "" {
		var err error
		certPEM, err = ioutil.ReadFile(certFile)
		if err != nil {
			t.Fatal(err)
		}
	}
	if keyFile != "" {
		var err error
		keyPEM, err = ioutil.ReadFile(keyFile)
		if err != nil {
			t.Fatal(err)
		}
	}

	obj := map[string]string{"flavor": flavor}
	if len(domain) > 0 {
		obj["domain"] = domain
	}
	if len(ip) > 0 {
		obj["ip"] = ip
	}
	if len(certPEM) > 0 {
		obj["certificate"] = string(certPEM)
	}
	if len(keyPEM) > 0 {
		obj["private_key"] = string(keyPEM)
	}

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

func TestNewHandler(t *testing.T) {
	newTestHandler(t)
}

type bundleTest struct {
	Domain             string
	IP                 string
	CertFile           string
	KeyFile            string
	Flavor             string
	ExpectedHTTPStatus int
	ExpectedSuccess    bool
	ExpectedErrorCode  int
}

var bundleTests = []bundleTest{
	// Test bundling with certificate
	{
		CertFile:           testLeafCertFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertFile:           testLeafCertFile,
		Flavor:             "ubiquitous",
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertFile:           testLeafCertFile,
		Flavor:             "optimal",
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertFile:           testLeafCertFile,
		KeyFile:            testLeafKeyFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertFile:           testLeafCertFile,
		Domain:             "cfssl-leaf.com",
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	// Test bundling with remote domain
	{
		Domain:             "google.com",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
	},
	// Error testing.
	{
		CertFile:           testLeafCertFile,
		KeyFile:            testLeafWrongKeyFile,
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  2300,
	},
	{
		// no input parameter is specified
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  http.StatusBadRequest,
	},
	{
		CertFile:           testBrokenCertFile,
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  1003,
	},
	{
		CertFile:           testLeafKeyFile,
		KeyFile:            testLeafKeyFile,
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  1003,
	},
	{
		CertFile:           testLeafCertFile,
		KeyFile:            testLeafCertFile,
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  2003,
	},
	{
		CertFile:           testLeafCertFile,
		Domain:             "cloudflare-leaf.com",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  1200,
	},
}

func TestBundle(t *testing.T) {
	for i, test := range bundleTests {
		resp, body := testBundleFile(t, test.Domain, test.IP, test.CertFile, test.KeyFile, test.Flavor)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Errorf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Errorf("failed to read response body: %v", err)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

		if test.ExpectedSuccess != message.Success {
			t.Errorf("Test %d: expected: %v, have %v", i, test.ExpectedSuccess, message.Success)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}
		if test.ExpectedSuccess == true {
			continue
		}

		if test.ExpectedErrorCode != 0 && test.ExpectedErrorCode != message.Errors[0].Code {
			t.Errorf("Test %d: expected: %v, have %v", i, test.ExpectedErrorCode, message.Errors[0].Code)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}
	}
}
