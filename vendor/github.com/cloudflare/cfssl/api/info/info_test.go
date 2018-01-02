package info

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

const (
	testCaFile    = "../testdata/ca.pem"
	testCaKeyFile = "../testdata/ca_key.pem"

	// second test CA for multiroot
	testCaFile2    = "../testdata/ca2.pem"
	testCaKeyFile2 = "../testdata/ca2-key.pem"
)

// Generally, the single root function and its multiroot analogue will
// be presented together.

func newTestHandler(t *testing.T) (h http.Handler) {
	signer, err := local.NewSignerFromFile(testCaFile, testCaKeyFile, nil)
	if err != nil {
		t.Fatal(err)
	}

	h, err = NewHandler(signer)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newTestMultiHandler(t *testing.T) (h http.Handler) {
	signer1, err := local.NewSignerFromFile(testCaFile, testCaKeyFile, nil)
	if err != nil {
		t.Fatal(err)
	}

	signer2, err := local.NewSignerFromFile(testCaFile2, testCaKeyFile2, nil)
	if err != nil {
		t.Fatal(err)
	}

	signers := map[string]signer.Signer{
		"test1": signer1,
		"test2": signer2,
	}

	h, err = NewMultiHandler(signers, "test1")
	if err != nil {
		t.Fatalf("%v", err)
	}

	return
}

func TestNewHandler(t *testing.T) {
	newTestHandler(t)
}

func TestNewMultiHandler(t *testing.T) {
	newTestMultiHandler(t)
}

func newInfoServer(t *testing.T) *httptest.Server {
	ts := httptest.NewServer(newTestHandler(t))
	return ts
}

func newMultiInfoServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(newTestMultiHandler(t))
}

func testInfoFile(t *testing.T, req map[string]interface{}) (resp *http.Response, body []byte) {
	ts := newInfoServer(t)
	defer ts.Close()

	blob, err := json.Marshal(req)
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

func testMultiInfoFile(t *testing.T, req map[string]interface{}) (resp *http.Response, body []byte) {
	ts := newMultiInfoServer(t)
	defer ts.Close()

	blob, err := json.Marshal(req)
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

type infoTest struct {
	RequestObject      map[string]interface{}
	ExpectedHTTPStatus int
	ExpectedSuccess    bool
	ExpectedErrorCode  int
}

var infoTests = []infoTest{
	{
		map[string]interface{}{
			"label":   "",
			"profile": "",
		},
		http.StatusOK,
		true,
		0,
	},
	{
		map[string]interface{}{
			"label": 123,
		},
		http.StatusBadRequest,
		false,
		http.StatusBadRequest,
	},
}

var multiInfoTests = []infoTest{
	{
		map[string]interface{}{
			"label":   "",
			"profile": "",
		},
		http.StatusOK,
		true,
		0,
	},
	{
		map[string]interface{}{
			"label":   "test1",
			"profile": "",
		},
		http.StatusOK,
		true,
		0,
	},
	{
		map[string]interface{}{
			"label":   "test2",
			"profile": "",
		},
		http.StatusOK,
		true,
		0,
	},
	{
		map[string]interface{}{
			"label":   "badlabel",
			"profile": "",
		},
		http.StatusBadRequest,
		false,
		http.StatusBadRequest,
	},
	{
		map[string]interface{}{
			"label": 123,
		},
		http.StatusBadRequest,
		false,
		http.StatusBadRequest,
	},
}

func TestInfo(t *testing.T) {
	for i, test := range infoTests {
		resp, body := testInfoFile(t, test.RequestObject)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Fatalf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Fatalf("failed to read response body: %v", err)
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

func TestMultiInfo(t *testing.T) {
	for i, test := range multiInfoTests {
		resp, body := testMultiInfoFile(t, test.RequestObject)
		if resp.StatusCode != test.ExpectedHTTPStatus {
			t.Fatalf("Test %d: expected: %d, have %d", i, test.ExpectedHTTPStatus, resp.StatusCode)
			t.Fatal(resp.Status, test.ExpectedHTTPStatus, string(body))
		}

		message := new(api.Response)
		err := json.Unmarshal(body, message)
		if err != nil {
			t.Fatalf("failed to read response body: %v", err)
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
