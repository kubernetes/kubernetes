package v4

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
)

func TestStripExcessHeaders(t *testing.T) {
	vals := []string{
		"",
		"123",
		"1 2 3",
		"1 2 3 ",
		"  1 2 3",
		"1  2 3",
		"1  23",
		"1  2  3",
		"1  2  ",
		" 1  2  ",
		"12   3",
		"12   3   1",
		"12           3     1",
		"12     3       1abc123",
	}

	expected := []string{
		"",
		"123",
		"1 2 3",
		"1 2 3",
		"1 2 3",
		"1 2 3",
		"1 23",
		"1 2 3",
		"1 2",
		"1 2",
		"12 3",
		"12 3 1",
		"12 3 1",
		"12 3 1abc123",
	}

	stripExcessSpaces(vals)
	for i := 0; i < len(vals); i++ {
		if e, a := expected[i], vals[i]; e != a {
			t.Errorf("%d, expect %v, got %v", i, e, a)
		}
	}
}

func buildRequest(serviceName, region, body string) (*http.Request, io.ReadSeeker) {
	endpoint := "https://" + serviceName + "." + region + ".amazonaws.com"
	reader := strings.NewReader(body)
	req, _ := http.NewRequest("POST", endpoint, reader)
	req.URL.Opaque = "//example.org/bucket/key-._~,!@#$%^&*()"
	req.Header.Add("X-Amz-Target", "prefix.Operation")
	req.Header.Add("Content-Type", "application/x-amz-json-1.0")
	req.Header.Add("Content-Length", string(len(body)))
	req.Header.Add("X-Amz-Meta-Other-Header", "some-value=!@#$%^&* (+)")
	req.Header.Add("X-Amz-Meta-Other-Header_With_Underscore", "some-value=!@#$%^&* (+)")
	req.Header.Add("X-amz-Meta-Other-Header_With_Underscore", "some-value=!@#$%^&* (+)")
	return req, reader
}

func buildSigner() Signer {
	return Signer{
		Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
	}
}

func removeWS(text string) string {
	text = strings.Replace(text, " ", "", -1)
	text = strings.Replace(text, "\n", "", -1)
	text = strings.Replace(text, "\t", "", -1)
	return text
}

func assertEqual(t *testing.T, expected, given string) {
	if removeWS(expected) != removeWS(given) {
		t.Errorf("\nExpected: %s\nGiven:    %s", expected, given)
	}
}

func TestPresignRequest(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "{}")

	signer := buildSigner()
	signer.Presign(req, body, "dynamodb", "us-east-1", 300*time.Second, time.Unix(0, 0))

	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-length;content-type;host;x-amz-meta-other-header;x-amz-meta-other-header_with_underscore"
	expectedSig := "ea7856749041f727690c580569738282e99c79355fe0d8f125d3b5535d2ece83"
	expectedCred := "AKID/19700101/us-east-1/dynamodb/aws4_request"
	expectedTarget := "prefix.Operation"

	q := req.URL.Query()
	if e, a := expectedSig, q.Get("X-Amz-Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedCred, q.Get("X-Amz-Credential"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedHeaders, q.Get("X-Amz-SignedHeaders"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedDate, q.Get("X-Amz-Date"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if a := q.Get("X-Amz-Meta-Other-Header"); len(a) != 0 {
		t.Errorf("expect %v to be empty", a)
	}
	if e, a := expectedTarget, q.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestPresignBodyWithArrayRequest(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	req.URL.RawQuery = "Foo=z&Foo=o&Foo=m&Foo=a"

	signer := buildSigner()
	signer.Presign(req, body, "dynamodb", "us-east-1", 300*time.Second, time.Unix(0, 0))

	expectedDate := "19700101T000000Z"
	expectedHeaders := "content-length;content-type;host;x-amz-meta-other-header;x-amz-meta-other-header_with_underscore"
	expectedSig := "fef6002062400bbf526d70f1a6456abc0fb2e213fe1416012737eebd42a62924"
	expectedCred := "AKID/19700101/us-east-1/dynamodb/aws4_request"
	expectedTarget := "prefix.Operation"

	q := req.URL.Query()
	if e, a := expectedSig, q.Get("X-Amz-Signature"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedCred, q.Get("X-Amz-Credential"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedHeaders, q.Get("X-Amz-SignedHeaders"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedDate, q.Get("X-Amz-Date"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if a := q.Get("X-Amz-Meta-Other-Header"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
	if e, a := expectedTarget, q.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignRequest(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	signer := buildSigner()
	signer.Sign(req, body, "dynamodb", "us-east-1", time.Unix(0, 0))

	expectedDate := "19700101T000000Z"
	expectedSig := "AWS4-HMAC-SHA256 Credential=AKID/19700101/us-east-1/dynamodb/aws4_request, SignedHeaders=content-length;content-type;host;x-amz-date;x-amz-meta-other-header;x-amz-meta-other-header_with_underscore;x-amz-security-token;x-amz-target, Signature=ea766cabd2ec977d955a3c2bae1ae54f4515d70752f2207618396f20aa85bd21"

	q := req.Header
	if e, a := expectedSig, q.Get("Authorization"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := expectedDate, q.Get("X-Amz-Date"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignBodyS3(t *testing.T) {
	req, body := buildRequest("s3", "us-east-1", "hello")
	signer := buildSigner()
	signer.Sign(req, body, "s3", "us-east-1", time.Now())
	hash := req.Header.Get("X-Amz-Content-Sha256")
	if e, a := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", hash; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignBodyGlacier(t *testing.T) {
	req, body := buildRequest("glacier", "us-east-1", "hello")
	signer := buildSigner()
	signer.Sign(req, body, "glacier", "us-east-1", time.Now())
	hash := req.Header.Get("X-Amz-Content-Sha256")
	if e, a := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", hash; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestPresignEmptyBodyS3(t *testing.T) {
	req, body := buildRequest("s3", "us-east-1", "hello")
	signer := buildSigner()
	signer.Presign(req, body, "s3", "us-east-1", 5*time.Minute, time.Now())
	hash := req.Header.Get("X-Amz-Content-Sha256")
	if e, a := "UNSIGNED-PAYLOAD", hash; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignPrecomputedBodyChecksum(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "hello")
	req.Header.Set("X-Amz-Content-Sha256", "PRECOMPUTED")
	signer := buildSigner()
	signer.Sign(req, body, "dynamodb", "us-east-1", time.Now())
	hash := req.Header.Get("X-Amz-Content-Sha256")
	if e, a := "PRECOMPUTED", hash; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestAnonymousCredentials(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{Credentials: credentials.AnonymousCredentials})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)
	SignSDKRequest(r)

	urlQ := r.HTTPRequest.URL.Query()
	if a := urlQ.Get("X-Amz-Signature"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
	if a := urlQ.Get("X-Amz-Credential"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
	if a := urlQ.Get("X-Amz-SignedHeaders"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
	if a := urlQ.Get("X-Amz-Date"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}

	hQ := r.HTTPRequest.Header
	if a := hQ.Get("Authorization"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
	if a := hQ.Get("X-Amz-Date"); len(a) != 0 {
		t.Errorf("expect %v to be empty, was not", a)
	}
}

func TestIgnoreResignRequestWithValidCreds(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{
		Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
		Region:      aws.String("us-west-2"),
	})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)

	SignSDKRequest(r)
	sig := r.HTTPRequest.Header.Get("Authorization")

	signSDKRequestWithCurrTime(r, func() time.Time {
		// Simulate one second has passed so that signature's date changes
		// when it is resigned.
		return time.Now().Add(1 * time.Second)
	})
	if e, a := sig, r.HTTPRequest.Header.Get("Authorization"); e == a {
		t.Errorf("expect %v to be %v, but was not", e, a)
	}
}

func TestIgnorePreResignRequestWithValidCreds(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{
		Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
		Region:      aws.String("us-west-2"),
	})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)
	r.ExpireTime = time.Minute * 10

	SignSDKRequest(r)
	sig := r.HTTPRequest.URL.Query().Get("X-Amz-Signature")

	signSDKRequestWithCurrTime(r, func() time.Time {
		// Simulate one second has passed so that signature's date changes
		// when it is resigned.
		return time.Now().Add(1 * time.Second)
	})
	if e, a := sig, r.HTTPRequest.URL.Query().Get("X-Amz-Signature"); e == a {
		t.Errorf("expect %v to be %v, but was not", e, a)
	}
}

func TestResignRequestExpiredCreds(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	svc := awstesting.NewClient(&aws.Config{Credentials: creds})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)
	SignSDKRequest(r)
	querySig := r.HTTPRequest.Header.Get("Authorization")
	var origSignedHeaders string
	for _, p := range strings.Split(querySig, ", ") {
		if strings.HasPrefix(p, "SignedHeaders=") {
			origSignedHeaders = p[len("SignedHeaders="):]
			break
		}
	}
	if a := origSignedHeaders; len(a) == 0 {
		t.Errorf("expect not to be empty, but was")
	}
	if e, a := origSignedHeaders, "authorization"; strings.Contains(a, e) {
		t.Errorf("expect %v to not be in %v, but was", e, a)
	}
	origSignedAt := r.LastSignedAt

	creds.Expire()

	signSDKRequestWithCurrTime(r, func() time.Time {
		// Simulate one second has passed so that signature's date changes
		// when it is resigned.
		return time.Now().Add(1 * time.Second)
	})
	updatedQuerySig := r.HTTPRequest.Header.Get("Authorization")
	if e, a := querySig, updatedQuerySig; e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}

	var updatedSignedHeaders string
	for _, p := range strings.Split(updatedQuerySig, ", ") {
		if strings.HasPrefix(p, "SignedHeaders=") {
			updatedSignedHeaders = p[len("SignedHeaders="):]
			break
		}
	}
	if a := updatedSignedHeaders; len(a) == 0 {
		t.Errorf("expect not to be empty, but was")
	}
	if e, a := updatedQuerySig, "authorization"; strings.Contains(a, e) {
		t.Errorf("expect %v to not be in %v, but was", e, a)
	}
	if e, a := origSignedAt, r.LastSignedAt; e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}
}

func TestPreResignRequestExpiredCreds(t *testing.T) {
	provider := &credentials.StaticProvider{Value: credentials.Value{
		AccessKeyID:     "AKID",
		SecretAccessKey: "SECRET",
		SessionToken:    "SESSION",
	}}
	creds := credentials.NewCredentials(provider)
	svc := awstesting.NewClient(&aws.Config{Credentials: creds})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)
	r.ExpireTime = time.Minute * 10

	SignSDKRequest(r)
	querySig := r.HTTPRequest.URL.Query().Get("X-Amz-Signature")
	signedHeaders := r.HTTPRequest.URL.Query().Get("X-Amz-SignedHeaders")
	if a := signedHeaders; len(a) == 0 {
		t.Errorf("expect not to be empty, but was")
	}
	origSignedAt := r.LastSignedAt

	creds.Expire()

	signSDKRequestWithCurrTime(r, func() time.Time {
		// Simulate the request occurred 15 minutes in the past
		return time.Now().Add(-48 * time.Hour)
	})
	if e, a := querySig, r.HTTPRequest.URL.Query().Get("X-Amz-Signature"); e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}
	resignedHeaders := r.HTTPRequest.URL.Query().Get("X-Amz-SignedHeaders")
	if e, a := signedHeaders, resignedHeaders; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := signedHeaders, "x-amz-signedHeaders"; strings.Contains(a, e) {
		t.Errorf("expect %v to not be in %v, but was", e, a)
	}
	if e, a := origSignedAt, r.LastSignedAt; e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}
}

func TestResignRequestExpiredRequest(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	svc := awstesting.NewClient(&aws.Config{Credentials: creds})
	r := svc.NewRequest(
		&request.Operation{
			Name:       "BatchGetItem",
			HTTPMethod: "POST",
			HTTPPath:   "/",
		},
		nil,
		nil,
	)

	SignSDKRequest(r)
	querySig := r.HTTPRequest.Header.Get("Authorization")
	origSignedAt := r.LastSignedAt

	signSDKRequestWithCurrTime(r, func() time.Time {
		// Simulate the request occurred 15 minutes in the past
		return time.Now().Add(15 * time.Minute)
	})
	if e, a := querySig, r.HTTPRequest.Header.Get("Authorization"); e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}
	if e, a := origSignedAt, r.LastSignedAt; e == a {
		t.Errorf("expect %v to be %v, was not", e, a)
	}
}

func TestSignWithRequestBody(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	signer := NewSigner(creds)

	expectBody := []byte("abc123")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := ioutil.ReadAll(r.Body)
		r.Body.Close()
		if err != nil {
			t.Errorf("expect no error, got %v", err)
		}
		if e, a := expectBody, b; !reflect.DeepEqual(e, a) {
			t.Errorf("expect %v, got %v", e, a)
		}
		w.WriteHeader(http.StatusOK)
	}))

	req, err := http.NewRequest("POST", server.URL, nil)

	_, err = signer.Sign(req, bytes.NewReader(expectBody), "service", "region", time.Now())
	if err != nil {
		t.Errorf("expect not no error, got %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Errorf("expect not no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignWithRequestBody_Overwrite(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	signer := NewSigner(creds)

	var expectBody []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := ioutil.ReadAll(r.Body)
		r.Body.Close()
		if err != nil {
			t.Errorf("expect not no error, got %v", err)
		}
		if e, a := len(expectBody), len(b); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		w.WriteHeader(http.StatusOK)
	}))

	req, err := http.NewRequest("GET", server.URL, strings.NewReader("invalid body"))

	_, err = signer.Sign(req, nil, "service", "region", time.Now())
	req.ContentLength = 0

	if err != nil {
		t.Errorf("expect not no error, got %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Errorf("expect not no error, got %v", err)
	}
	if e, a := http.StatusOK, resp.StatusCode; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestBuildCanonicalRequest(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	req.URL.RawQuery = "Foo=z&Foo=o&Foo=m&Foo=a"
	ctx := &signingCtx{
		ServiceName: "dynamodb",
		Region:      "us-east-1",
		Request:     req,
		Body:        body,
		Query:       req.URL.Query(),
		Time:        time.Now(),
		ExpireTime:  5 * time.Second,
	}

	ctx.buildCanonicalString()
	expected := "https://example.org/bucket/key-._~,!@#$%^&*()?Foo=z&Foo=o&Foo=m&Foo=a"
	if e, a := expected, ctx.Request.URL.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignWithBody_ReplaceRequestBody(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	req, seekerBody := buildRequest("dynamodb", "us-east-1", "{}")
	req.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))

	s := NewSigner(creds)
	origBody := req.Body

	_, err := s.Sign(req, seekerBody, "dynamodb", "us-east-1", time.Now())
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if req.Body == origBody {
		t.Errorf("expeect request body to not be origBody")
	}

	if req.Body == nil {
		t.Errorf("expect request body to be changed but was nil")
	}
}

func TestSignWithBody_NoReplaceRequestBody(t *testing.T) {
	creds := credentials.NewStaticCredentials("AKID", "SECRET", "SESSION")
	req, seekerBody := buildRequest("dynamodb", "us-east-1", "{}")
	req.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))

	s := NewSigner(creds, func(signer *Signer) {
		signer.DisableRequestBodyOverwrite = true
	})

	origBody := req.Body

	_, err := s.Sign(req, seekerBody, "dynamodb", "us-east-1", time.Now())
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if req.Body != origBody {
		t.Errorf("expect request body to not be chagned")
	}
}

func TestRequestHost(t *testing.T) {
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	req.URL.RawQuery = "Foo=z&Foo=o&Foo=m&Foo=a"
	req.Host = "myhost"
	ctx := &signingCtx{
		ServiceName: "dynamodb",
		Region:      "us-east-1",
		Request:     req,
		Body:        body,
		Query:       req.URL.Query(),
		Time:        time.Now(),
		ExpireTime:  5 * time.Second,
	}

	ctx.buildCanonicalHeaders(ignoredHeaders, ctx.Request.Header)
	if !strings.Contains(ctx.canonicalHeaders, "host:"+req.Host) {
		t.Errorf("canonical host header invalid")
	}
}

func BenchmarkPresignRequest(b *testing.B) {
	signer := buildSigner()
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	for i := 0; i < b.N; i++ {
		signer.Presign(req, body, "dynamodb", "us-east-1", 300*time.Second, time.Now())
	}
}

func BenchmarkSignRequest(b *testing.B) {
	signer := buildSigner()
	req, body := buildRequest("dynamodb", "us-east-1", "{}")
	for i := 0; i < b.N; i++ {
		signer.Sign(req, body, "dynamodb", "us-east-1", time.Now())
	}
}

var stripExcessSpaceCases = []string{
	`AWS4-HMAC-SHA256 Credential=AKIDFAKEIDFAKEID/20160628/us-west-2/s3/aws4_request, SignedHeaders=host;x-amz-date, Signature=1234567890abcdef1234567890abcdef1234567890abcdef`,
	`123   321   123   321`,
	`   123   321   123   321   `,
	`   123    321    123          321   `,
	"123",
	"1 2 3",
	"  1 2 3",
	"1  2 3",
	"1  23",
	"1  2  3",
	"1  2  ",
	" 1  2  ",
	"12   3",
	"12   3   1",
	"12           3     1",
	"12     3       1abc123",
}

func BenchmarkStripExcessSpaces(b *testing.B) {
	for i := 0; i < b.N; i++ {
		// Make sure to start with a copy of the cases
		cases := append([]string{}, stripExcessSpaceCases...)
		stripExcessSpaces(cases)
	}
}
