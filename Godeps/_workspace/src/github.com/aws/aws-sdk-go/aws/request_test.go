package aws

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/internal/apierr"
	"github.com/stretchr/testify/assert"
)

type testData struct {
	Data string
}

func body(str string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(str)))
}

func unmarshal(req *Request) {
	defer req.HTTPResponse.Body.Close()
	if req.Data != nil {
		json.NewDecoder(req.HTTPResponse.Body).Decode(req.Data)
	}
	return
}

func unmarshalError(req *Request) {
	bodyBytes, err := ioutil.ReadAll(req.HTTPResponse.Body)
	if err != nil {
		req.Error = apierr.New("UnmarshaleError", req.HTTPResponse.Status, err)
		return
	}
	if len(bodyBytes) == 0 {
		req.Error = apierr.NewRequestError(
			apierr.New("UnmarshaleError", req.HTTPResponse.Status, fmt.Errorf("empty body")),
			req.HTTPResponse.StatusCode,
			"",
		)
		return
	}
	var jsonErr jsonErrorResponse
	if err := json.Unmarshal(bodyBytes, &jsonErr); err != nil {
		req.Error = apierr.New("UnmarshaleError", "JSON unmarshal", err)
		return
	}
	req.Error = apierr.NewRequestError(
		apierr.New(jsonErr.Code, jsonErr.Message, nil),
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
		http.Response{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		http.Response{StatusCode: 501, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		http.Response{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := NewService(&Config{MaxRetries: 10})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := NewRequest(s, &Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	assert.Nil(t, err)
	assert.Equal(t, 2, int(r.RetryCount))
	assert.Equal(t, "valid", out.Data)
}

// test that retries occur for 4xx status codes with a response type that can be retried - see `shouldRetry`
func TestRequestRecoverRetry4xxRetryable(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		http.Response{StatusCode: 400, Body: body(`{"__type":"Throttling","message":"Rate exceeded."}`)},
		http.Response{StatusCode: 429, Body: body(`{"__type":"ProvisionedThroughputExceededException","message":"Rate exceeded."}`)},
		http.Response{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := NewService(&Config{MaxRetries: 10})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := NewRequest(s, &Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	assert.Nil(t, err)
	assert.Equal(t, 2, int(r.RetryCount))
	assert.Equal(t, "valid", out.Data)
}

// test that retries don't occur for 4xx status codes with a response type that can't be retried
func TestRequest4xxUnretryable(t *testing.T) {
	s := NewService(&Config{MaxRetries: 10})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *Request) {
		r.HTTPResponse = &http.Response{StatusCode: 401, Body: body(`{"__type":"SignatureDoesNotMatch","message":"Signature does not match."}`)}
	})
	out := &testData{}
	r := NewRequest(s, &Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	assert.NotNil(t, err)
	if e, ok := err.(awserr.RequestFailure); ok {
		assert.Equal(t, 401, e.StatusCode())
	} else {
		assert.Fail(t, "Expected error to be a service failure")
	}
	assert.Equal(t, "SignatureDoesNotMatch", err.(awserr.Error).Code())
	assert.Equal(t, "Signature does not match.", err.(awserr.Error).Message())
	assert.Equal(t, 0, int(r.RetryCount))
}

func TestRequestExhaustRetries(t *testing.T) {
	delays := []time.Duration{}
	sleepDelay = func(delay time.Duration) {
		delays = append(delays, delay)
	}

	reqNum := 0
	reqs := []http.Response{
		http.Response{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		http.Response{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		http.Response{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
		http.Response{StatusCode: 500, Body: body(`{"__type":"UnknownError","message":"An error occurred."}`)},
	}

	s := NewService(&Config{MaxRetries: -1})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	r := NewRequest(s, &Operation{Name: "Operation"}, nil, nil)
	err := r.Send()
	assert.NotNil(t, err)
	if e, ok := err.(awserr.RequestFailure); ok {
		assert.Equal(t, 500, e.StatusCode())
	} else {
		assert.Fail(t, "Expected error to be a service failure")
	}
	assert.Equal(t, "UnknownError", err.(awserr.Error).Code())
	assert.Equal(t, "An error occurred.", err.(awserr.Error).Message())
	assert.Equal(t, 3, int(r.RetryCount))
	assert.True(t, reflect.DeepEqual([]time.Duration{30 * time.Millisecond, 60 * time.Millisecond, 120 * time.Millisecond}, delays))
}

// test that the request is retried after the credentials are expired.
func TestRequestRecoverExpiredCreds(t *testing.T) {
	reqNum := 0
	reqs := []http.Response{
		http.Response{StatusCode: 400, Body: body(`{"__type":"ExpiredTokenException","message":"expired token"}`)},
		http.Response{StatusCode: 200, Body: body(`{"data":"valid"}`)},
	}

	s := NewService(&Config{MaxRetries: 10, Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "")})
	s.Handlers.Validate.Clear()
	s.Handlers.Unmarshal.PushBack(unmarshal)
	s.Handlers.UnmarshalError.PushBack(unmarshalError)

	credExpiredBeforeRetry := false
	credExpiredAfterRetry := false

	s.Handlers.AfterRetry.PushBack(func(r *Request) {
		credExpiredAfterRetry = r.Config.Credentials.IsExpired()
	})

	s.Handlers.Sign.Clear()
	s.Handlers.Sign.PushBack(func(r *Request) {
		r.Config.Credentials.Get()
	})
	s.Handlers.Send.Clear() // mock sending
	s.Handlers.Send.PushBack(func(r *Request) {
		r.HTTPResponse = &reqs[reqNum]
		reqNum++
	})
	out := &testData{}
	r := NewRequest(s, &Operation{Name: "Operation"}, nil, out)
	err := r.Send()
	assert.Nil(t, err)

	assert.False(t, credExpiredBeforeRetry, "Expect valid creds before retry check")
	assert.True(t, credExpiredAfterRetry, "Expect expired creds after retry check")
	assert.False(t, s.Config.Credentials.IsExpired(), "Expect valid creds after cred expired recovery")

	assert.Equal(t, 1, int(r.RetryCount))
	assert.Equal(t, "valid", out.Data)
}
