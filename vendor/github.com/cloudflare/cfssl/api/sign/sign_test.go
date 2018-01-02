package sign

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/signer"
)

const (
	testCaFile         = "../testdata/ca.pem"
	testCaKeyFile      = "../testdata/ca_key.pem"
	testCSRFile        = "../testdata/csr.pem"
	testBrokenCertFile = "../testdata/broken.pem"
	testBrokenCSRFile  = "../testdata/broken_csr.pem"
)

var validLocalConfig = `
{
	"signing": {
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "1m"
		}
	}
}`

var validAuthLocalConfig = `
{
	"signing": {
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "1m",
			"auth_key": "sample"
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	}
}`

var validMixedLocalConfig = `
{
	"signing": {
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "1m"
		},
		"profiles": {
			"auth": {
				"usages": ["digital signature", "email protection"],
				"expiry": "1m",
				"auth_key": "sample"
			}
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	}
}`

var alsoValidMixedLocalConfig = `
{
	"signing": {
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "1m",
			"auth_key": "sample"
		},
		"profiles": {
			"no-auth": {
				"usages": ["digital signature", "email protection"],
				"expiry": "1m"
			}
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	}
}`

func newTestHandler(t *testing.T) (h http.Handler) {
	h, err := NewHandler(testCaFile, testCaKeyFile, nil)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestNewHandler(t *testing.T) {
	newTestHandler(t)
}

func TestNewHandlerWithProfile(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNewHandlerWithAuthProfile(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validAuthLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err == nil {
		t.Fatal("All profiles have auth keys. Should have failed to create non-auth sign handler.")
	}
}

func TestNewHandlerError(t *testing.T) {
	// using testBrokenCSRFile as badly formed key
	_, err := NewHandler(testCaFile, testBrokenCSRFile, nil)
	if err == nil {
		t.Fatal("Expect error when create a signer with broken file.")
	}
}

func TestNewAuthHandlerWithNonAuthProfile(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewAuthHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err == nil {
		t.Fatal("No profile have auth keys. Should have failed to create auth sign handler.")
	}
}

func TestNewHandlersWithMixedProfile(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validMixedLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal("Should be able to create non-auth sign handler.")
	}

	_, err = NewAuthHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal("Should be able to create auth sign handler.")
	}
}

func TestNewHandlersWithAnotherMixedProfile(t *testing.T) {
	conf, err := config.LoadConfig([]byte(alsoValidMixedLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal("Should be able to create non-auth sign handler.")
	}

	_, err = NewAuthHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal("Should be able to create auth sign handler.")
	}
}

func newSignServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestHandler(t))
	return ts
}

func testSignFileOldInterface(t *testing.T, hostname, csrFile string) (resp *http.Response, body []byte) {
	ts := newSignServer(t)
	defer ts.Close()
	var csrPEM []byte
	if csrFile != "" {
		var err error
		csrPEM, err = ioutil.ReadFile(csrFile)
		if err != nil {
			t.Fatal(err)
		}
	}
	obj := map[string]string{}
	if len(hostname) > 0 {
		obj["hostname"] = hostname
	}
	if len(csrPEM) > 0 {
		obj["certificate_request"] = string(csrPEM)
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

func testSignFile(t *testing.T, hosts []string, subject *signer.Subject, csrFile string) (resp *http.Response, body []byte) {
	ts := newSignServer(t)
	defer ts.Close()
	var csrPEM []byte
	if csrFile != "" {
		var err error
		csrPEM, err = ioutil.ReadFile(csrFile)
		if err != nil {
			t.Fatal(err)
		}
	}
	obj := map[string]interface{}{}
	if hosts != nil {
		obj["hosts"] = hosts
	}
	if len(csrPEM) > 0 {
		obj["certificate_request"] = string(csrPEM)
	}
	if subject != nil {
		obj["subject"] = subject
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

const (
	testHostName   = "localhost"
	testDomainName = "cloudflare.com"
)

type signTest struct {
	Hosts              []string
	Subject            *signer.Subject
	CSRFile            string
	ExpectedHTTPStatus int
	ExpectedSuccess    bool
	ExpectedErrorCode  int
}

var signTests = []signTest{
	{
		Hosts:              []string{testHostName},
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              []string{testDomainName},
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              []string{testDomainName, testHostName},
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              []string{testDomainName},
		Subject:            &signer.Subject{CN: "example.com"},
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              []string{},
		Subject:            &signer.Subject{CN: "example.com"},
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              nil,
		CSRFile:            testCSRFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		Hosts:              []string{testDomainName},
		CSRFile:            "",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  http.StatusBadRequest,
	},
	{
		Hosts:              []string{testDomainName},
		CSRFile:            testBrokenCSRFile,
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  9002,
	},
}

func TestSign(t *testing.T) {
	for i, test := range signTests {
		resp, body := testSignFile(t, test.Hosts, test.Subject, test.CSRFile)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Logf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Logf("failed to read response body: %v", err)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

		if test.ExpectedSuccess != message.Success {
			t.Logf("Test %d: expected: %v, have %v", i, test.ExpectedSuccess, message.Success)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}
		if test.ExpectedSuccess == true {
			continue
		}

		if test.ExpectedErrorCode != message.Errors[0].Code {
			t.Fatalf("Test %d: expected: %v, have %v", i, test.ExpectedErrorCode, message.Errors[0].Code)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

	}

	// Test for backward compatibility
	// TODO remove after API transition is complete.
	for i, test := range signTests {
		// an empty hostname is not accepted by the old interface but an empty hosts array should be accepted
		// so skip the case of empty hosts array for the old interface.
		if test.Hosts != nil && len(test.Hosts) == 0 {
			continue
		}

		hostname := strings.Join(test.Hosts, ",")
		resp, body := testSignFileOldInterface(t, hostname, test.CSRFile)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Logf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Logf("failed to read response body: %v", err)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

		if test.ExpectedSuccess != message.Success {
			t.Logf("Test %d: expected: %v, have %v", i, test.ExpectedSuccess, message.Success)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}
		if test.ExpectedSuccess == true {
			continue
		}

		if test.ExpectedErrorCode != message.Errors[0].Code {
			t.Fatalf("Test %d: expected: %v, have %v", i, test.ExpectedErrorCode, message.Errors[0].Code)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

	}

}

func newTestAuthHandler(t *testing.T) http.Handler {
	conf, err := config.LoadConfig([]byte(validAuthLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	h, err := NewAuthHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func TestNewAuthHandler(t *testing.T) {
	newTestAuthHandler(t)
}

func TestNewAuthHandlerWithNoAuthConfig(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validLocalConfig))
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewAuthHandler(testCaFile, testCaKeyFile, conf.Signing)
	if err == nil {
		t.Fatal("Config doesn't have auth keys. Should have failed.")
	}
	return
}

func testAuthSignFile(t *testing.T, hosts []string, subject *signer.Subject, csrFile string, profile *config.SigningProfile) (resp *http.Response, body []byte) {
	ts := newAuthSignServer(t)
	defer ts.Close()

	var csrPEM []byte
	if csrFile != "" {
		var err error
		csrPEM, err = ioutil.ReadFile(csrFile)
		if err != nil {
			t.Fatal(err)
		}
	}
	obj := map[string]interface{}{}
	if hosts != nil {
		obj["hosts"] = hosts
	}
	if subject != nil {
		obj["subject"] = subject
	}
	if len(csrPEM) > 0 {
		obj["certificate_request"] = string(csrPEM)
	}

	reqBlob, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}

	var aReq auth.AuthenticatedRequest
	aReq.Request = reqBlob
	aReq.Token, err = profile.Provider.Token(aReq.Request)
	if err != nil {
		t.Fatal(err)
	}

	blob, err := json.Marshal(aReq)
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

func newAuthSignServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestAuthHandler(t))
	return ts
}

func TestAuthSign(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validAuthLocalConfig))
	if err != nil {
		t.Fatal(err)
	}
	for i, test := range signTests {
		resp, body := testAuthSignFile(t, test.Hosts, test.Subject, test.CSRFile, conf.Signing.Default)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Logf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Logf("failed to read response body: %v", err)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

		if test.ExpectedSuccess != message.Success {
			t.Fatalf("Test %d: expected: %v, have %v", i, test.ExpectedSuccess, message.Success)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}
		if test.ExpectedSuccess == true {
			continue
		}

		if test.ExpectedErrorCode != message.Errors[0].Code {
			t.Fatalf("Test %d: expected: %v, have %v", i, test.ExpectedErrorCode, message.Errors[0].Code)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
		}

	}
}
