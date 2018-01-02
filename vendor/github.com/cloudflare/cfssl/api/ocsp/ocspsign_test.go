package ocsp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/ocsp"
	goocsp "golang.org/x/crypto/ocsp"

	"github.com/cloudflare/cfssl/helpers"
)

const (
	testCaFile       = "../../ocsp/testdata/ca.pem"
	testRespCertFile = "../../ocsp/testdata/server.crt"
	testKeyFile      = "../../ocsp/testdata/server.key"
	testCertFile     = "../../ocsp/testdata/cert.pem"
)

func newTestHandler(t *testing.T) http.Handler {
	// arbitrary duration
	dur, _ := time.ParseDuration("1ms")
	s, err := ocsp.NewSignerFromFile(testCaFile, testRespCertFile, testKeyFile, dur)
	if err != nil {
		t.Fatalf("Signer creation failed %v", err)
	}
	return NewHandler(s)
}

func TestNewHandler(t *testing.T) {
	newTestHandler(t)
}

func newSignServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestHandler(t))
	return ts
}

func testSignFile(t *testing.T, certFile, status string, reason int, revokedAt string) (resp *http.Response, body []byte) {
	ts := newSignServer(t)
	defer ts.Close()

	obj := map[string]interface{}{}
	if certFile != "" {
		c, err := ioutil.ReadFile(certFile)
		if err != nil {
			t.Fatal(err)
		}
		obj["certificate"] = string(c)
	}
	if status != "" {
		obj["status"] = status
	}
	obj["reason"] = reason
	if revokedAt != "" {
		obj["revoked_at"] = revokedAt
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

type signTest struct {
	CertificateFile    string
	Status             string
	Reason             int
	RevokedAt          string
	ExpectedHTTPStatus int
	ExpectedSuccess    bool
	ExpectedErrorCode  int
}

var signTests = []signTest{
	{
		CertificateFile:    testCertFile,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertificateFile:    testCertFile,
		Status:             "revoked",
		Reason:             1,
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertificateFile:    testCertFile,
		Status:             "revoked",
		RevokedAt:          "now",
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertificateFile:    testCertFile,
		Status:             "revoked",
		RevokedAt:          "2015-08-15",
		ExpectedHTTPStatus: http.StatusOK,
		ExpectedSuccess:    true,
		ExpectedErrorCode:  0,
	},
	{
		CertificateFile:    testCertFile,
		Status:             "revoked",
		RevokedAt:          "a",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  http.StatusBadRequest,
	},
	{
		CertificateFile:    "",
		Status:             "",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  http.StatusBadRequest,
	},
	{
		CertificateFile:    testCertFile,
		Status:             "_",
		ExpectedHTTPStatus: http.StatusBadRequest,
		ExpectedSuccess:    false,
		ExpectedErrorCode:  8200,
	},
}

func TestSign(t *testing.T) {
	for i, test := range signTests {
		resp, body := testSignFile(t, test.CertificateFile, test.Status, test.Reason, test.RevokedAt)
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
		if !test.ExpectedSuccess {
			if test.ExpectedErrorCode != message.Errors[0].Code {
				t.Fatalf("Test %d: expected: %v, have %v", i, test.ExpectedErrorCode, message.Errors[0].Code)
				t.Fatal(resp.Status, test.ExpectedHTTPStatus, message)
			}
			continue
		}

		result, ok := message.Result.(map[string]interface{})
		if !ok {
			t.Logf("failed to read result")
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}
		b64Resp, ok := result["ocspResponse"].(string)
		if !ok {
			t.Logf("failed to find ocspResponse")
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		der, err := base64.StdEncoding.DecodeString(b64Resp)
		if err != nil {
			t.Logf("failed to decode base64")
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, b64Resp)
		}

		ocspResp, err := goocsp.ParseResponse(der, nil)
		if err != nil {
			t.Logf("failed to parse ocsp response: %v", err)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, b64Resp)
		}

		//should default to good
		if test.Status == "" {
			test.Status = "good"
		}
		intStatus := ocsp.StatusCode[test.Status]
		if ocspResp.Status != intStatus {
			t.Fatalf("Test %d incorrect status: expected: %v, have %v", i, intStatus, ocspResp.Status)
			t.Fatal(ocspResp.Status, intStatus, ocspResp)
		}

		if test.Status == "revoked" {
			if ocspResp.RevocationReason != test.Reason {
				t.Fatalf("Test %d incorrect reason: expected: %v, have %v", i, test.Reason, ocspResp.RevocationReason)
				t.Fatal(ocspResp.RevocationReason, test.Reason, ocspResp)
			}

			var r time.Time
			if test.RevokedAt == "" || test.RevokedAt == "now" {
				r = time.Now().UTC().Truncate(helpers.OneDay)
			} else {
				r, _ = time.Parse("2006-01-02", test.RevokedAt)
			}

			if !ocspResp.RevokedAt.Truncate(helpers.OneDay).Equal(r) {
				t.Fatalf("Test %d incorrect revokedAt: expected: %v, have %v", i, r, ocspResp.RevokedAt)
				t.Fatal(ocspResp.RevokedAt, test.RevokedAt, ocspResp)
			}
		}
	}
}
