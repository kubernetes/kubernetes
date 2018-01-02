package request_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
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
	return
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
		{StatusCode: 501, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
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
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 2, int(r.RetryCount); e != a {
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
		{StatusCode: 429, Body: body(`{"__type":"ProvisionedThroughputExceededException","message":"Rate exceeded."}`)},
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
	if err != nil {
		t.Fatalf("expect no error, but got %v", err)
	}
	if e, a := 2, int(r.RetryCount); e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}
	if e, a := "valid", out.Data; e != a {
		t.Errorf("expect %q output got %q", e, a)
	}
}

// test that retries don't occur for 4xx status codes with a response type that can't be retried
func TestRequest4xxUnretryable(t *testing.T) {
	s := awstesting.NewClient(aws.NewConfig().WithMaxRetries(10))
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{StatusCode: 401, Body: body(`{"__type":"SignatureDoesNotMatch","message":"Signature does not match."}`)}
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
	if e, a := 0, int(r.RetryCount); e != a {
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

	s := awstesting.NewClient(aws.NewConfig().WithSleepDelay(sleepDelay))
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
	if e, a := 3, int(r.RetryCount); e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}

	expectDelays := []struct{ min, max time.Duration }{{30, 59}, {60, 118}, {120, 236}}
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

	s := awstesting.NewClient(&aws.Config{MaxRetries: aws.Int(10), Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "")})
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

	if e, a := 1, int(r.RetryCount); e != a {
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

	if e, a := "foo/bar name/version (extra1; extra2)", r.HTTPRequest.Header.Get("User-Agent"); e != a {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestMakeAddtoUserAgentFreeFormHandler(t *testing.T) {
	fn := request.MakeAddToUserAgentFreeFormHandler("name/version (extra1; extra2)")
	r := &request.Request{HTTPRequest: &http.Request{Header: http.Header{}}}
	r.HTTPRequest.Header.Set("User-Agent", "foo/bar")
	fn(r)

	if e, a := "foo/bar name/version (extra1; extra2)", r.HTTPRequest.Header.Get("User-Agent"); e != a {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestRequestUserAgent(t *testing.T) {
	s := awstesting.NewClient(&aws.Config{Region: aws.String("us-east-1")})
	//	s.Handlers.Validate.Clear()

	req := s.NewRequest(&request.Operation{Name: "Operation"}, nil, &testData{})
	req.HTTPRequest.Header.Set("User-Agent", "foo/bar")
	if err := req.Build(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectUA := fmt.Sprintf("foo/bar %s/%s (%s; %s; %s)",
		aws.SDKName, aws.SDKVersion, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	if e, a := expectUA, req.HTTPRequest.Header.Get("User-Agent"); e != a {
		t.Errorf("expect %q user agent, got %q", e, a)
	}
}

func TestRequestThrottleRetries(t *testing.T) {
	delays := []time.Duration{}
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

	s := awstesting.NewClient(aws.NewConfig().WithSleepDelay(sleepDelay))
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
	if e, a := 3, int(r.RetryCount); e != a {
		t.Errorf("expect %d retry count, got %d", e, a)
	}

	expectDelays := []struct{ min, max time.Duration }{{500, 999}, {1000, 1998}, {2000, 3996}}
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
	if e, a := 1, int(r.RetryCount); e != a {
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
	if e, a := 1, int(r.RetryCount); e != a {
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

		if err := r.Send(); err != nil {
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
			err:      awserr.New(request.ErrCodeSerialization, "foo error", stubConnectionResetError),
			expected: true,
		},
	}

	for i, c := range testCases {
		r := &request.Request{
			Error: c.err,
		}
		if r.IsErrorRetryable() != c.expected {
			t.Errorf("Case %d: Expected %v, but received %v", i+1, c.expected, !c.expected)
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

type connResetCloser struct {
}

func (rc *connResetCloser) Read(b []byte) (int, error) {
	return 0, stubConnectionResetError
}

func (rc *connResetCloser) Close() error {
	return nil
}

func TestSerializationErrConnectionReset(t *testing.T) {
	count := 0
	handlers := request.Handlers{}
	handlers.Send.PushBack(func(r *request.Request) {
		count++
		r.HTTPResponse = &http.Response{}
		r.HTTPResponse.Body = &connResetCloser{}
	})

	handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)
	handlers.AfterRetry.PushBackNamed(corehandlers.AfterRetryHandler)

	op := &request.Operation{
		Name:       "op",
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	meta := metadata.ClientInfo{
		ServiceName:   "fooService",
		SigningName:   "foo",
		SigningRegion: "foo",
		Endpoint:      "localhost",
		APIVersion:    "2001-01-01",
		JSONVersion:   "1.1",
		TargetPrefix:  "Foo",
	}
	cfg := unit.Session.Config.Copy()
	cfg.MaxRetries = aws.Int(5)

	req := request.New(
		*cfg,
		meta,
		handlers,
		client.DefaultRetryer{NumMaxRetries: 5},
		op,
		&struct {
		}{},
		&struct {
		}{},
	)

	osErr := stubConnectionResetError
	req.ApplyOptions(request.WithResponseReadTimeout(time.Second))
	err := req.Send()
	if err == nil {
		t.Error("Expected rror 'SerializationError', but received nil")
	}
	if aerr, ok := err.(awserr.Error); ok && aerr.Code() != "SerializationError" {
		t.Errorf("Expected 'SerializationError', but received %q", aerr.Code())
	} else if !ok {
		t.Errorf("Expected 'awserr.Error', but received %v", reflect.TypeOf(err))
	} else if aerr.OrigErr().Error() != osErr.Error() {
		t.Errorf("Expected %q, but received %q", osErr.Error(), aerr.OrigErr().Error())
	}

	if count != 6 {
		t.Errorf("Expected '6', but received %d", count)
	}
}

type testRetryer struct {
	shouldRetry bool
}

func (d *testRetryer) MaxRetries() int {
	return 3
}

// RetryRules returns the delay duration before retrying this request again
func (d *testRetryer) RetryRules(r *request.Request) time.Duration {
	return time.Duration(time.Millisecond)
}

func (d *testRetryer) ShouldRetry(r *request.Request) bool {
	d.shouldRetry = true
	if r.Retryable != nil {
		return *r.Retryable
	}

	if r.HTTPResponse.StatusCode >= 500 {
		return true
	}
	return r.IsErrorRetryable()
}

func TestEnforceShouldRetryCheck(t *testing.T) {
	tp := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		ResponseHeaderTimeout: 1 * time.Millisecond,
	}

	client := &http.Client{Transport: tp}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This server should wait forever. Requests will timeout and the SDK should
		// attempt to retry.
		select {}
	}))

	retryer := &testRetryer{}
	s := awstesting.NewClient(&aws.Config{
		Region:                  aws.String("mock-region"),
		MaxRetries:              aws.Int(0),
		Endpoint:                aws.String(server.URL),
		DisableSSL:              aws.Bool(true),
		Retryer:                 retryer,
		HTTPClient:              client,
		EnforceShouldRetryCheck: aws.Bool(true),
	})

	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)

	out := &testData{}
	r := s.NewRequest(&request.Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	if err == nil {
		t.Fatalf("expect error, but got nil")
	}
	if e, a := 3, int(r.RetryCount); e != a {
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
