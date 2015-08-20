package auth

import (
	"encoding/json"
	"io/ioutil"
	"testing"
)

var (
	testProvider   Provider
	testProviderAD Provider
	testKey        = "0123456789ABCDEF0123456789ABCDEF"
	testAD         = []byte{1, 2, 3, 4} // IP address 1.2.3.4
)

func TestNew(t *testing.T) {
	_, err := New("ABC", nil)
	if err == nil {
		t.Fatal("expected failure with improperly-hex-encoded key")
	}

	testProvider, err = New(testKey, nil)
	if err != nil {
		t.Fatalf("%v", err)
	}

	testProviderAD, err = New(testKey, testAD)
	if err != nil {
		t.Fatalf("%v", err)
	}

}

var (
	testRequest1A = &AuthenticatedRequest{
		Request: []byte(`testing 1 2 3`),
	}
	testRequest1B = &AuthenticatedRequest{
		Request: []byte(`testing 1 2 3`),
	}
	testRequest2 = &AuthenticatedRequest{
		Request: []byte(`testing 3 2 1`),
	}
)

// Sanity check: can a newly-generated token be verified?
func TestVerifyTrue(t *testing.T) {
	var err error

	testRequest1A.Token, err = testProvider.Token(testRequest1A.Request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	testRequest1B.Token, err = testProviderAD.Token(testRequest1B.Request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if !testProvider.Verify(testRequest1A) {
		t.Fatal("failed to verify request 1A")
	}

	if !testProviderAD.Verify(testRequest1B) {
		t.Fatal("failed to verify request 1B")
	}
}

// Sanity check: ensure that additional data is actually used in
// verification.
func TestVerifyAD(t *testing.T) {
	if testProvider.Verify(testRequest1B) {
		t.Fatal("no-AD provider verifies request with AD")
	}

	if testProviderAD.Verify(testRequest1A) {
		t.Fatal("AD provider verifies request without AD")
	}
}

// Sanity check: verification fails if tokens are not the same length.
func TestTokenLength(t *testing.T) {
	token := testRequest1A.Token[:]
	testRequest1A.Token = testRequest1A.Token[1:]

	if testProvider.Verify(testRequest1A) {
		t.Fatal("invalid token should not be verified")
	}

	testRequest1A.Token = token
}

// Sanity check: token fails validation if the request is changed.
func TestBadRequest(t *testing.T) {
	testRequest2.Token = testRequest1A.Token
	if testProvider.Verify(testRequest2) {
		t.Fatal("bad request should fail verification")
	}
}

// Sanity check: a null request should fail to verify.
func TestNullRequest(t *testing.T) {
	if testProvider.Verify(nil) {
		t.Fatal("null request should fail verification")
	}
}

// Sanity check: verify a pre-generated authenticated request.
func TestPreGenerated(t *testing.T) {
	in, err := ioutil.ReadFile("testdata/authrequest.json")
	if err != nil {
		t.Fatalf("%v", err)
	}

	var req AuthenticatedRequest
	err = json.Unmarshal(in, &req)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if !testProvider.Verify(&req) {
		t.Fatal("failed to verify pre-generated request")
	}
}

var bmRequest []byte

func TestLoadBenchmarkRequest(t *testing.T) {
	in, err := ioutil.ReadFile("testdata/request.json")
	if err != nil {
		t.Fatalf("%v", err)
	}

	bmRequest = in
}

func BenchmarkToken(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := testProvider.Token(bmRequest)
		if err != nil {
			b.Fatalf("%v", err)
		}
	}
}

func BenchmarkVerify(b *testing.B) {
	token, _ := testProvider.Token(bmRequest)
	req := &AuthenticatedRequest{
		Token:   token,
		Request: bmRequest,
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if !testProvider.Verify(req) {
			b.Fatal("failed to verify request")
		}
	}
}
