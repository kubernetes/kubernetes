package request_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

type tempNetworkError struct {
	op     string
	msg    string
	isTemp bool
}

func (e *tempNetworkError) Temporary() bool { return e.isTemp }
func (e *tempNetworkError) Error() string {
	return fmt.Sprintf("%s: %s", e.op, e.msg)
}

var (
	// net.OpError accept, are always temporary
	errAcceptConnectionResetStub = &tempNetworkError{
		isTemp: true, op: "accept", msg: "connection reset",
	}

	// net.OpError read for ECONNRESET is not temporary.
	errReadConnectionResetStub = &tempNetworkError{
		isTemp: false, op: "read", msg: "connection reset",
	}

	// net.OpError write for ECONNRESET may not be temporary, but is treaded as
	// temporary by the SDK.
	errWriteConnectionResetStub = &tempNetworkError{
		isTemp: false, op: "write", msg: "connection reset",
	}

	// net.OpError write for broken pipe may not be temporary, but is treaded as
	// temporary by the SDK.
	errWriteBrokenPipeStub = &tempNetworkError{
		isTemp: false, op: "write", msg: "broken pipe",
	}

	// Generic connection reset error
	errConnectionResetStub = errors.New("connection reset")

	// use of closed network connection error
	errUseOfClosedConnectionStub = errors.New("use of closed network connection")
)

type testData struct {
	Data string
}

func body(str string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(str)))
}

func unmarshal(req *request.Request) {
	defer req.HTTPResponse.Body.Close()
	if req.Data != nil {
		json.NewDecoder(req.HTTPResponse.Body).Decode(req.Data)
	}
}

func unmarshalError(req *request.Request) {
	bodyBytes, err := ioutil.ReadAll(req.HTTPResponse.Body)
	if err != nil {
		req.Error = awserr.New("UnmarshaleError", req.HTTPResponse.Status, err)
		return
	}
	if len(bodyBytes) == 0 {
		req.Error = awserr.NewRequestFailure(
			awserr.New("UnmarshaleError", req.HTTPResponse.Status, fmt.Errorf("empty body")),
			req.HTTPResponse.StatusCode,
			"",
		)
		return
	}
	var jsonErr jsonErrorResponse
	if err := json.Unmarshal(bodyBytes, &jsonErr); err != nil {
		req.Error = awserr.New("UnmarshaleError", "JSON unmarshal", err)
		return
	}
	req.Error = awserr.NewRequestFailure(
		awserr.New(jsonErr.Code, jsonErr.Message, nil),
		req.HTTPResponse.StatusCode,
		"",
	)
}

type jsonErrorResponse struct {
	Code    string `json:"__type"`
	Message string `json:"message"`
}

// test that retries occur for 5xx status codes
func TestRequestRecoverRetry5xx(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 502, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := awstesting.NewClient(&aws.Config{
		MaxRetries: aws.Int(10),
		SleepDelay: func(time.Duration) {},
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 2, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

// test that retries occur for 4xx status codes with a response type that can be retried - see `shouldRetry`
func TestRequestRecoverRetry4xxRetryable(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 400, Body: body(`{"__type":"Throttling","message":"Rate exceeded."}`)},
		{StatusCode: 400, Body: body(`{"__type":"ProvisionedThroughputExceededException","message":"Rate exceeded."}`)},
		{StatusCode: 429, Body: body(`{"__type":"FooException","message":"Rate exceeded."}`)},
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := awstesting.NewClient(&aws.Config{
		MaxRetries: aws.Int(10),
		SleepDelay: func(time.Duration) {},
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 3, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

// test that retries don't occur for 4xx status codes with a response type that can't be retried
func TestRequest4xxUnretryable(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		MaxRetries: aws.Int(1),
		SleepDelay: func(time.Duration) {},
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: 401,
			Body:       body(`{"__type":"SignatureDoesNotMatch","message":"Signature does not match."}`),
		}
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err == nil {
		t.Fatalf("expect error, but did not get one")
	}
	aerr := err.(awserr.RequestFailure)
	if e, a := 401, aerr.StatusCode(); e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
	if e, a := "SignatureDoesNotMatch", aerr.Code(); e != a {
		t.Errorf("expect %q error code, got %q", e, a)
	}
	if e, a := "Signature does not match.", aerr.Message(); e != a {
		t.Errorf("expect %q error message, got %q", e, a)
	}
	if e, a := 0, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
}

func TestRequestExhaustRetries(t *testing.T) {
	delays := []time.Duration{}
	sleepDelay := func(delay time.Duration) {
		delays = append(delays, delay)
	}

	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
	}

	s := awstesting.NewClient(&aws.Config{
		SleepDelay: sleepDelay,
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)
	err := r.Send()
	if err == nil {
		t.Fatalf("expect error, but did not get one")
	}
	aerr := err.(awserr.RequestFailure)
	if e, a := 500, aerr.StatusCode(); e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
	if e, a := "UnknownError", aerr.Code(); e != a {
		t.Errorf("expect %q error code, got %q", e, a)
	}
	if e, a := "An error occurred.", aerr.Message(); e != a {
		t.Errorf("expect %q error message, got %q", e, a)
	}
	if e, a := 3, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}

	expectDelays := []struct{ min, max time.Duration }{{30, 60}, {60, 120}, {120, 240}}
	for i, v := range delays {
		min := expectDelays[i].min * time.Millisecond
		max := expectDelays[i].max * time.Millisecond
		if !(min <= v && v <= max) {
			t.Errorf("Expect delay to be within range, i:%d, v:%s, min:%s, max:%s",
				i, v, min, max)
		}
	}
}

// test that the request is retried after the credentials are expired.
func TestRequestRecoverExpiredCreds(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 400, Body: body(`{"__type":"ExpiredTokenException","message":"expired token"}`)},
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := awstesting.NewClient(&aws.Config{
		MaxRetries:  aws.Int(10),
		Credentials: credentials.NewStaticCredentials("AKID", "SECRET", ""),
		SleepDelay:  func(time.Duration) {},
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)

	credExpiredBeforeRetry := false
	credExpiredAfterRetry := false

	s.Handlers.AfterRetry.PushBack(func(r *request.Request) {
		credExpiredAfterRetry = r.Config.Credentials.IsExpired()
	})

	s.Handlers.Sign.Clear()
	s.Handlers.Sign.PushBack(func(r *request.Request) {
		r.Config.Credentials.Get()
	})
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if credExpiredBeforeRetry {
		t.Errorf("Expect valid creds before retry check")
	}
	if !credExpiredAfterRetry {
		t.Errorf("Expect expired creds after retry check")
	}
	if s.Config.Credentials.IsExpired() {
		t.Errorf("Expect valid creds after cred expired recovery")
	}

	if e, a := 1, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

func TestMakeAddtoUserAgentHandler(t *testing.T) {
	fn := request.MakeAddToUserAgentHandler("name", "version", "extra1", "extra2")
	r := &request.Request{HTTPRequest: &http.Request{Header: http.Header{}}}
	r.HTTPRequest.Header.Set("User-Agent", "foo/bar")
	fn(r)

	if e, a := "foo/bar name/version (extra1; extra2)", r.HTTPRequest.Header.Get("User-Agent"); !strings.HasPrefix(a, e) {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestMakeAddtoUserAgentFreeFormHandler(t *testing.T) {
	fn := request.MakeAddToUserAgentFreeFormHandler("name/version (extra1; extra2)")
	r := &request.Request{HTTPRequest: &http.Request{Header: http.Header{}}}
	r.HTTPRequest.Header.Set("User-Agent", "foo/bar")
	fn(r)

	if e, a := "foo/bar name/version (extra1; extra2)", r.HTTPRequest.Header.Get("User-Agent"); !strings.HasPrefix(a, e) {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestRequestUserAgent(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		Region: aws.String("us-east-1"),
	})

	req := s.NewRequest(&request.Operation{Name: "Operation"}, nil, &testData{})
	req.HTTPRequest.Header.Set("User-Agent", "foo/bar")
	if err := req.Build(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectUA := fmt.Sprintf("foo/bar %s/%s (%s; %s; %s)",
		aws.SDKName, aws.SDKVersion, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	if e, a := expectUA, req.HTTPRequest.Header.Get("User-Agent"); !strings.HasPrefix(a, e) {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestRequestThrottleRetries(t *testing.T) {
	var delays []time.Duration
	sleepDelay := func(delay time.Duration) {
		delays = append(delays, delay)
	}

	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 500, Body: body(`{"__type":"Throttling","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"Throttling","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"Throttling","message":"An error occurred."}`)},
		{StatusCode: 500, Body: body(`{"__type":"Throttling","message":"An error occurred."}`)},
	}

	s := awstesting.NewClient(&aws.Config{
		SleepDelay: sleepDelay,
	})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)
	err := r.Send()
	if err == nil {
		t.Fatalf("expect error, but did not get one")
	}
	aerr := err.(awserr.RequestFailure)
	if e, a := 500, aerr.StatusCode(); e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
	if e, a := "Throttling", aerr.Code(); e != a {
		t.Errorf("expect %q error code, got %q", e, a)
	}
	if e, a := "An error occurred.", aerr.Message(); e != a {
		t.Errorf("expect %q error message, got %q", e, a)
	}
	if e, a := 3, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}

	expectDelays := []struct{ min, max time.Duration }{{500, 1000}, {1000, 2000}, {2000, 4000}}
	for i, v := range delays {
		min := expectDelays[i].min * time.Millisecond
		max := expectDelays[i].max * time.Millisecond
		if !(min <= v && v <= max) {
			t.Errorf("Expect delay to be within range, i:%d, v:%s, min:%s, max:%s",
				i, v, min, max)
		}
	}
}

// test that retries occur for request timeouts when response.Body can be nil
func TestRequestRecoverTimeoutWithNilBody(t *testing.T) {
	reqNum := 0
	reqs := []*http.Response{
		{StatusCode: 0, Body: nil}, // body can be nil when requests time out
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}
	errors := []error{
		errTimeout, nil,
	}

	s := awstesting.NewClient(aws.NewConfig().WithMaxRetries(10))
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.AfterRetry.Clear() // force retry on all errors
	s.Handlers.AfterRetry.PushBack(func(r *request.Request) {
		if r.Error != nil {
			r.Error = nil
			r.Retryable = aws.Bool(true)
			r.RetryCount++
		}
	})
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = reqs[reqNum]
		r.Error = errors[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 1, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

func TestRequestRecoverTimeoutWithNilResponse(t *testing.T) {
	reqNum := 0
	reqs := []*http.Response{
		nil,
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}
	errors := []error{
		errTimeout,
		nil,
	}

	s := awstesting.NewClient(aws.NewConfig().WithMaxRetries(10))
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.AfterRetry.Clear() // force retry on all errors
	s.Handlers.AfterRetry.PushBack(func(r *request.Request) {
		if r.Error != nil {
			r.Error = nil
			r.Retryable = aws.Bool(true)
			r.RetryCount++
		}
	})
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = reqs[reqNum]
		r.Error = errors[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 1, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

func TestRequest_NoBody(t *testing.T) {
	cases := []string{
		"GET", "HEAD", "DELETE",
		"PUT", "POST", "PATCH",
	}

	for i, c := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if v := r.TransferEncoding; len(v) > 0 {
				t.Errorf("%d, expect no body sent with Transfer-Encoding, %v", i, v)
			}

			outMsg := []byte(`{"Value": "abc"}`)

			if b, err := ioutil.ReadAll(r.Body); err != nil {
				t.Fatalf("%d, expect no error reading request body, got %v", i, err)
			} else if n := len(b); n > 0 {
				t.Errorf("%d, expect no request body, got %d bytes", i, n)
			}

			w.Header().Set("Content-Length", strconv.Itoa(len(outMsg)))
			if _, err := w.Write(outMsg); err != nil {
				t.Fatalf("%d, expect no error writing server response, got %v", i, err)
			}
		}))

		s := awstesting.NewClient(&aws.Config{
			Region:     aws.String("mock-region"),
			MaxRetries: aws.Int(0),
			Endpoint:   aws.String(server.URL),
			DisableSSL: aws.Bool(true),
		})
		s.Handlers.Build.PushBack(rest.Build)
		s.Handlers.Validate.Clear()
		s.Handlers.Unmarshal.PushBack(unmarshal)
		s.Handlers.UnmarshalError.PushBack(unmarshalError)

		in := struct {
			Bucket *string `location:"uri" locationName:"bucket"`
			Key    *string `location:"uri" locationName:"key"`
		}{
			Bucket: aws.String("mybucket"), Key: aws.String("myKey"),
		}

		out := struct {
			Value *string
		}{}

		r := s.NewRequest(&request.Operation{
			Name: "OpName", HTTPMethod: c, HTTPPath: "/{bucket}/{key+}",
		}, &in, &out)

		err := r.Send()
		server.Close()
		if err != nil {
			t.Fatalf("%d, expect no error sending request, got %v", i, err)
		}
	}
}

func TestIsSerializationErrorRetryable(t *testing.T) {
	testCases := []struct {
		err      error
		expected bool
	}{
		{
			err:      awserr.New(request.ErrCodeSerialization, "foo error", nil),
			expected: false,
		},
		{
			err:      awserr.New("ErrFoo", "foo error", nil),
			expected: false,
		},
		{
			err:      nil,
			expected: false,
		},
		{
			err:      awserr.New(request.ErrCodeSerialization, "foo error", errAcceptConnectionResetStub),
			expected: true,
		},
	}

	for i, c := range testCases {
		r := &request.Request{
			Error: c.err,
		}
		if r.IsErrorRetryable() != c.expected {
			t.Errorf("Case %d: Expected %v, but received %v", i, c.expected, !c.expected)
		}
	}
}

func TestWithLogLevel(t *testing.T) {
	r := &request.Request{}

	opt := request.WithLogLevel(aws.LogDebugWithHTTPBody)
	r.ApplyOptions(opt)

	if !r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody) {
		t.Errorf("expect log level to be set, but was not, %v",
			r.Config.LogLevel.Value())
	}
}

func TestWithGetResponseHeader(t *testing.T) {
	r := &request.Request{}

	var val, val2 string
	r.ApplyOptions(
		request.WithGetResponseHeader("x-a-header", &val),
		request.WithGetResponseHeader("x-second-header", &val2),
	)

	r.HTTPResponse = &http.Response{
		Header: func() http.Header {
			h := http.Header{}
			h.Set("x-a-header", "first")
			h.Set("x-second-header", "second")
			return h
		}(),
	}
	r.Handlers.Complete.Run(r)

	if e, a := "first", val; e != a {
		t.Errorf("expect %q header value got %q", e, a)
	}
	if e, a := "second", val2; e != a {
		t.Errorf("expect %q header value got %q", e, a)
	}
}

func TestWithGetResponseHeaders(t *testing.T) {
	r := &request.Request{}

	var headers http.Header
	opt := request.WithGetResponseHeaders(&headers)

	r.ApplyOptions(opt)

	r.HTTPResponse = &http.Response{
		Header: func() http.Header {
			h := http.Header{}
			h.Set("x-a-header", "headerValue")
			return h
		}(),
	}
	r.Handlers.Complete.Run(r)

	if e, a := "headerValue", headers.Get("x-a-header"); e != a {
		t.Errorf("expect %q header value got %q", e, a)
	}
}

type testRetryer struct {
	shouldRetry bool
	maxRetries  int
}

func (d *testRetryer) MaxRetries() int {
	return d.maxRetries
}

// RetryRules returns the delay duration before retrying this request again
func (d *testRetryer) RetryRules(r *request.Request) time.Duration {
	return 0
}

func (d *testRetryer) ShouldRetry(r *request.Request) bool {
	return d.shouldRetry
}

func TestEnforceShouldRetryCheck(t *testing.T) {

	retryer := &testRetryer{
		shouldRetry: true, maxRetries: 3,
	}
	s := awstesting.NewClient(&aws.Config{
		Region:                  aws.String("mock-region"),
		MaxRetries:              aws.Int(0),
		Retryer:                 retryer,
		EnforceShouldRetryCheck: aws.Bool(true),
		SleepDelay:              func(time.Duration) {},
	})

	s.Handlers.Validate.Clear()
	s.Handlers.Send.Swap(corehandlers.SendHandler.Name, request.NamedHandler{
		Name: "TestEnforceShouldRetryCheck",
		Fn: func(r *request.Request) {
			r.HTTPResponse = &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(bytes.NewBuffer(nil)),
			}
			r.Retryable = aws.Bool(false)
		},
	})

	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)

	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err == nil {
		t.Fatalf("expect error, but got nil")
	}
	if e, a := 3, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if !retryer.shouldRetry {
		t.Errorf("expect 'true' for ShouldRetry, but got %v", retryer.shouldRetry)
	}
}

type errReader struct {
	err error
}

func (reader *errReader) Read(b []byte) (int, error) {
	return 0, reader.err
}

func (reader *errReader) Close() error {
	return nil
}

func TestIsNoBodyReader(t *testing.T) {
	cases := []struct {
		reader io.ReadCloser
		expect bool
	}{
		{ioutil.NopCloser(bytes.NewReader([]byte("abc"))), false},
		{ioutil.NopCloser(bytes.NewReader(nil)), false},
		{nil, false},
		{request.NoBody, true},
	}

	for i, c := range cases {
		if e, a := c.expect, request.NoBody == c.reader; e != a {
			t.Errorf("%d, expect %t match, but was %t", i, e, a)
		}
	}
}

func TestRequest_TemporaryRetry(t *testing.T) {
	done := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1024")
		w.WriteHeader(http.StatusOK)

		w.Write(make([]byte, 100))

		f := w.(http.Flusher)
		f.Flush()

		<-done
	}))
	defer server.Close()

	client := &http.Client{
		Timeout: 100 * time.Millisecond,
	}

	svc := awstesting.NewClient(&aws.Config{
		Region:     unit.Session.Config.Region,
		MaxRetries: aws.Int(1),
		HTTPClient: client,
		DisableSSL: aws.Bool(true),
		Endpoint:   aws.String(server.URL),
	})

	req := svc.NewRequest(&request.Operation{
		Name: "name", HTTPMethod: "GET", HTTPPath: "/path",
	}, &struct{}{}, &struct{}{})

	req.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		defer req.HTTPResponse.Body.Close()
		_, err := io.Copy(ioutil.Discard, req.HTTPResponse.Body)
		r.Error = awserr.New(request.ErrCodeSerialization, "error", err)
	})

	err := req.Send()
	if err == nil {
		t.Errorf("expect error, got none")
	}
	close(done)

	aerr := err.(awserr.Error)
	if e, a := request.ErrCodeSerialization, aerr.Code(); e != a {
		t.Errorf("expect %q error code, got %q", e, a)
	}

	if e, a := 1, req.RetryCount; e != a {
		t.Errorf("expect %d retries, got %d", e, a)
	}

	type temporary interface {
		Temporary() bool
	}

	terr := aerr.OrigErr().(temporary)
	if !terr.Temporary() {
		t.Errorf("expect temporary error, was not")
	}
}

func TestRequest_Presign(t *testing.T) {
	presign := func(r *request.Request, expire time.Duration) (string, http.Header, error) {
		u, err := r.Presign(expire)
		return u, nil, err
	}
	presignRequest := func(r *request.Request, expire time.Duration) (string, http.Header, error) {
		return r.PresignRequest(expire)
	}
	mustParseURL := func(v string) *url.URL {
		u, err := url.Parse(v)
		if err != nil {
			panic(err)
		}
		return u
	}

	cases := []struct {
		Expire    time.Duration
		PresignFn func(*request.Request, time.Duration) (string, http.Header, error)
		SignerFn  func(*request.Request)
		URL       string
		Header    http.Header
		Err       string
	}{
		{
			PresignFn: presign,
			Err:       request.ErrCodeInvalidPresignExpire,
		},
		{
			PresignFn: presignRequest,
			Err:       request.ErrCodeInvalidPresignExpire,
		},
		{
			Expire:    -1,
			PresignFn: presign,
			Err:       request.ErrCodeInvalidPresignExpire,
		},
		{
			// Presign clear NotHoist
			Expire: 1 * time.Minute,
			PresignFn: func(r *request.Request, dur time.Duration) (string, http.Header, error) {
				r.NotHoist = true
				return presign(r, dur)
			},
			SignerFn: func(r *request.Request) {
				r.HTTPRequest.URL = mustParseURL("https://endpoint/presignedURL")
				if r.NotHoist {
					r.Error = fmt.Errorf("expect NotHoist to be cleared")
				}
			},
			URL: "https://endpoint/presignedURL",
		},
		{
			// PresignRequest does not clear NotHoist
			Expire: 1 * time.Minute,
			PresignFn: func(r *request.Request, dur time.Duration) (string, http.Header, error) {
				r.NotHoist = true
				return presignRequest(r, dur)
			},
			SignerFn: func(r *request.Request) {
				r.HTTPRequest.URL = mustParseURL("https://endpoint/presignedURL")
				if !r.NotHoist {
					r.Error = fmt.Errorf("expect NotHoist not to be cleared")
				}
			},
			URL: "https://endpoint/presignedURL",
		},
		{
			// PresignRequest returns signed headers
			Expire:    1 * time.Minute,
			PresignFn: presignRequest,
			SignerFn: func(r *request.Request) {
				r.HTTPRequest.URL = mustParseURL("https://endpoint/presignedURL")
				r.HTTPRequest.Header.Set("UnsigndHeader", "abc")
				r.SignedHeaderVals = http.Header{
					"X-Amzn-Header":  []string{"abc", "123"},
					"X-Amzn-Header2": []string{"efg", "456"},
				}
			},
			URL: "https://endpoint/presignedURL",
			Header: http.Header{
				"X-Amzn-Header":  []string{"abc", "123"},
				"X-Amzn-Header2": []string{"efg", "456"},
			},
		},
	}

	svc := awstesting.NewClient()
	svc.Handlers.Clear()
	for i, c := range cases {
		req := svc.NewRequest(&request.Operation{
			Name: "name", HTTPMethod: "GET", HTTPPath: "/path",
		}, &struct{}{}, &struct{}{})
		req.Handlers.Sign.PushBack(c.SignerFn)

		u, h, err := c.PresignFn(req, c.Expire)
		if len(c.Err) != 0 {
			if e, a := c.Err, err.Error(); !strings.Contains(a, e) {
				t.Errorf("%d, expect %v to be in %v", i, e, a)
			}
			continue
		} else if err != nil {
			t.Errorf("%d, expect no error, got %v", i, err)
			continue
		}
		if e, a := c.URL, u; e != a {
			t.Errorf("%d, expect %v URL, got %v", i, e, a)
		}
		if e, a := c.Header, h; !reflect.DeepEqual(e, a) {
			t.Errorf("%d, expect %v header got %v", i, e, a)
		}
	}
}

func TestSanitizeHostForHeader(t *testing.T) {
	cases := []struct {
		url                 string
		expectedRequestHost string
	}{
		{"https://estest.us-east-1.es.amazonaws.com:443", "estest.us-east-1.es.amazonaws.com"},
		{"https://estest.us-east-1.es.amazonaws.com", "estest.us-east-1.es.amazonaws.com"},
		{"https://localhost:9200", "localhost:9200"},
		{"http://localhost:80", "localhost"},
		{"http://localhost:8080", "localhost:8080"},
	}

	for _, c := range cases {
		r, _ := http.NewRequest("GET", c.url, nil)
		request.SanitizeHostForHeader(r)

		if h := r.Host; h != c.expectedRequestHost {
			t.Errorf("expect %v host, got %q", c.expectedRequestHost, h)
		}
	}
}

func TestRequestWillRetry_ByBody(t *testing.T) {
	svc := awstesting.NewClient()

	cases := []struct {
		WillRetry   bool
		HTTPMethod  string
		Body        io.ReadSeeker
		IsReqNoBody bool
	}{
		{
			WillRetry:   true,
			HTTPMethod:  "GET",
			Body:        bytes.NewReader([]byte{}),
			IsReqNoBody: true,
		},
		{
			WillRetry:   true,
			HTTPMethod:  "GET",
			Body:        bytes.NewReader(nil),
			IsReqNoBody: true,
		},
		{
			WillRetry:  true,
			HTTPMethod: "POST",
			Body:       bytes.NewReader([]byte("abc123")),
		},
		{
			WillRetry:  true,
			HTTPMethod: "POST",
			Body:       aws.ReadSeekCloser(bytes.NewReader([]byte("abc123"))),
		},
		{
			WillRetry:   true,
			HTTPMethod:  "GET",
			Body:        aws.ReadSeekCloser(bytes.NewBuffer(nil)),
			IsReqNoBody: true,
		},
		{
			WillRetry:   true,
			HTTPMethod:  "POST",
			Body:        aws.ReadSeekCloser(bytes.NewBuffer(nil)),
			IsReqNoBody: true,
		},
		{
			WillRetry:  false,
			HTTPMethod: "POST",
			Body:       aws.ReadSeekCloser(bytes.NewBuffer([]byte("abc123"))),
		},
	}

	for i, c := range cases {
		req := svc.NewRequest(&request.Operation{
			Name:       "Operation",
			HTTPMethod: c.HTTPMethod,
			HTTPPath:   "/",
		}, nil, nil)
		req.SetReaderBody(c.Body)
		req.Build()

		req.Error = fmt.Errorf("some error")
		req.Retryable = aws.Bool(true)
		req.HTTPResponse = &http.Response{
			StatusCode: 500,
		}

		if e, a := c.IsReqNoBody, request.NoBody == req.HTTPRequest.Body; e != a {
			t.Errorf("%d, expect request to be no body, %t, got %t, %T", i, e, a, req.HTTPRequest.Body)
		}

		if e, a := c.WillRetry, req.WillRetry(); e != a {
			t.Errorf("%d, expect %t willRetry, got %t", i, e, a)
		}

		if req.Error == nil {
			t.Fatalf("%d, expect error, got none", i)
		}
		if e, a := "some error", req.Error.Error(); !strings.Contains(a, e) {
			t.Errorf("%d, expect %q error in %q", i, e, a)
		}
		if e, a := 0, req.RetryCount; e != a {
			t.Errorf("%d, expect retry count to be %d, got %d", i, e, a)
		}
	}
}

func Test501NotRetrying(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		{StatusCode: 501, Body: body(`{"__type":"NotImplemented","message":"An error occurred."}`)},
		{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := awstesting.NewClient(aws.NewConfig().WithMaxRetries(10))
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err == nil {
		t.Fatal("expect error, but got none")
	}

	aerr := err.(awserr.Error)
	if e, a := "NotImplemented", aerr.Code(); e != a {
		t.Errorf("expected error code %q, but received %q", e, a)
	}
	if e, a := 1, r.RetryCount; e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
}

func TestRequestNoConnection(t *testing.T) {
	port, err := getFreePort()
	if err != nil {
		t.Fatalf("failed to get free port for test")
	}
	s := awstesting.NewClient(aws.NewConfig().
		WithMaxRetries(10).
		WithEndpoint("https://localhost:" + strconv.Itoa(port)).
		WithSleepDelay(func(time.Duration) {}),
	)
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)

	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)

	if err = r.Send(); err == nil {
		t.Fatal("expect error, but got none")
	}

	t.Logf("Error, %v", err)
	awsError := err.(awserr.Error)
	origError := awsError.OrigErr()
	t.Logf("Orig Error: %#v of type %T", origError, origError)

	if e, a := 10, r.RetryCount; e != a {
		t.Errorf("expect %v retry count, got %v", e, a)
	}
}

func TestRequestBodySeekFails(t *testing.T) {
	s := awstesting.NewClient()
	s.Handlers.Validate.Clear()
	s.Handlers.Build.Clear()

	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	r.SetReaderBody(&stubSeekFail{
		Err: fmt.Errorf("failed to seek reader"),
	})
	err := r.Send()
	if err == nil {
		t.Fatal("expect error, but got none")
	}

	aerr := err.(awserr.Error)
	if e, a := request.ErrCodeSerialization, aerr.Code(); e != a {
		t.Errorf("expect %v error code, got %v", e, a)
	}

}

func TestRequestEndpointWithDefaultPort(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		Endpoint: aws.String("https://example.test:443"),
	})
	r := s.NewRequest(&request.Operation{
		Name:       "FooBar",
		HTTPMethod: "GET",
		HTTPPath:   "/",
	}, nil, nil)
	r.Handlers.Validate.Clear()
	r.Handlers.ValidateResponse.Clear()
	r.Handlers.Send.Clear()
	r.Handlers.Send.PushFront(func(r *request.Request) {
		req := r.HTTPRequest

		if e, a := "example.test", req.Host; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

		if e, a := "https://example.test:443/", req.URL.String(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	})
	err := r.Send()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestRequestEndpointWithNonDefaultPort(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		Endpoint: aws.String("https://example.test:8443"),
	})
	r := s.NewRequest(&request.Operation{
		Name:       "FooBar",
		HTTPMethod: "GET",
		HTTPPath:   "/",
	}, nil, nil)
	r.Handlers.Validate.Clear()
	r.Handlers.ValidateResponse.Clear()
	r.Handlers.Send.Clear()
	r.Handlers.Send.PushFront(func(r *request.Request) {
		req := r.HTTPRequest

		// http.Request.Host should not be set for non-default ports
		if e, a := "", req.Host; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

		if e, a := "https://example.test:8443/", req.URL.String(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	})
	err := r.Send()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestRequestMarshaledEndpointWithDefaultPort(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		Endpoint: aws.String("https://example.test:443"),
	})
	r := s.NewRequest(&request.Operation{
		Name:       "FooBar",
		HTTPMethod: "GET",
		HTTPPath:   "/",
	}, nil, nil)
	r.Handlers.Validate.Clear()
	r.Handlers.ValidateResponse.Clear()
	r.Handlers.Build.PushBack(func(r *request.Request) {
		req := r.HTTPRequest
		req.URL.Host = "foo." + req.URL.Host
	})
	r.Handlers.Send.Clear()
	r.Handlers.Send.PushFront(func(r *request.Request) {
		req := r.HTTPRequest

		if e, a := "foo.example.test", req.Host; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

		if e, a := "https://foo.example.test:443/", req.URL.String(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	})
	err := r.Send()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestRequestMarshaledEndpointWithNonDefaultPort(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{
		Endpoint: aws.String("https://example.test:8443"),
	})
	r := s.NewRequest(&request.Operation{
		Name:       "FooBar",
		HTTPMethod: "GET",
		HTTPPath:   "/",
	}, nil, nil)
	r.Handlers.Validate.Clear()
	r.Handlers.ValidateResponse.Clear()
	r.Handlers.Build.PushBack(func(r *request.Request) {
		req := r.HTTPRequest
		req.URL.Host = "foo." + req.URL.Host
	})
	r.Handlers.Send.Clear()
	r.Handlers.Send.PushFront(func(r *request.Request) {
		req := r.HTTPRequest

		// http.Request.Host should not be set for non-default ports
		if e, a := "", req.Host; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

		if e, a := "https://foo.example.test:8443/", req.URL.String(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	})
	err := r.Send()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

type stubSeekFail struct {
	Err error
}

func (f *stubSeekFail) Read(b []byte) (int, error) {
	return len(b), nil
}
func (f *stubSeekFail) ReadAt(b []byte, offset int64) (int, error) {
	return len(b), nil
}
func (f *stubSeekFail) Seek(offset int64, mode int) (int64, error) {
	return 0, f.Err
}

func getFreePort() (int, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	defer l.Close()

	strAddr := l.Addr().String()
	parts := strings.Split(strAddr, ":")
	strPort := parts[len(parts)-1]
	port, err := strconv.ParseInt(strPort, 10, 32)
	if err != nil {
		return 0, err
	}
	return int(port), nil
}
