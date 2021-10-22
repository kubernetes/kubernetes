package request

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

func newRequest(t *testing.T, url string) *http.Request {
	r, err := http.NewRequest("GET", url, nil)
	if err != nil {
		t.Fatalf("can't forge request: %v", err)
	}
	return r
}

func TestShouldRetryError_nil(t *testing.T) {
	if shouldRetryError(nil) != true {
		t.Error("shouldRetryError(nil) should return true")
	}
}

func TestShouldRetryError_timeout(t *testing.T) {

	tr := &http.Transport{}
	defer tr.CloseIdleConnections()
	client := http.Client{
		Timeout:   time.Nanosecond,
		Transport: tr,
	}

	resp, err := client.Do(newRequest(t, "https://179.179.179.179/no/such/host"))
	if resp != nil {
		resp.Body.Close()
	}
	if err == nil {
		t.Fatal("This should have failed.")
	}
	debugerr(t, err)

	if shouldRetryError(err) == false {
		t.Errorf("this request timed out and should be retried")
	}
}

func TestShouldRetryError_cancelled(t *testing.T) {
	tr := &http.Transport{}
	defer tr.CloseIdleConnections()
	client := http.Client{
		Transport: tr,
	}

	cancelWait := make(chan bool)
	srvrWait := make(chan bool)
	srvr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		close(cancelWait) // Trigger the request cancel.
		time.Sleep(100 * time.Millisecond)

		fmt.Fprintf(w, "Hello")
		w.(http.Flusher).Flush() // send headers and some body
		<-srvrWait               // block forever
	}))
	defer srvr.Close()
	defer close(srvrWait)

	r := newRequest(t, srvr.URL)
	ch := make(chan struct{})
	r.Cancel = ch

	// Ensure the request has started, and client has started to receive bytes.
	// This ensures the test is stable and does not run into timing with the
	// request being canceled, before or after the http request is made.
	go func() {
		<-cancelWait
		close(ch) // request is cancelled before anything
	}()

	resp, err := client.Do(r)
	if resp != nil {
		resp.Body.Close()
	}
	if err == nil {
		t.Fatal("This should have failed.")
	}

	debugerr(t, err)

	if shouldRetryError(err) == true {
		t.Errorf("this request was cancelled and should not be retried")
	}
}

func TestShouldRetry(t *testing.T) {

	syscallError := os.SyscallError{
		Err:     ErrInvalidParams{},
		Syscall: "open",
	}

	opError := net.OpError{
		Op:     "dial",
		Net:    "tcp",
		Source: net.Addr(nil),
		Err:    &syscallError,
	}

	urlError := url.Error{
		Op:  "Post",
		URL: "https://localhost:52398",
		Err: &opError,
	}
	origError := awserr.New("ErrorTestShouldRetry", "Test should retry when error received", &urlError).OrigErr()
	if e, a := true, shouldRetryError(origError); e != a {
		t.Errorf("Expected to return %v to retry when error occurred, got %v instead", e, a)
	}

}

func debugerr(t *testing.T, err error) {
	t.Logf("Error, %v", err)

	switch err := err.(type) {
	case temporary:
		t.Logf("%s is a temporary error: %t", err, err.Temporary())
		return
	case *url.Error:
		t.Logf("err: %s, nested err: %#v", err, err.Err)
		if operr, ok := err.Err.(*net.OpError); ok {
			t.Logf("operr: %#v", operr)
		}
		debugerr(t, err.Err)
		return
	default:
		return
	}
}

func TestRequest_retryCustomCodes(t *testing.T) {
	cases := map[string]struct {
		Code               string
		RetryErrorCodes    []string
		ThrottleErrorCodes []string
		Retryable          bool
		Throttle           bool
	}{
		"retry code": {
			Code: "RetryMePlease",
			RetryErrorCodes: []string{
				"RetryMePlease",
				"SomeOtherError",
			},
			Retryable: true,
		},
		"throttle code": {
			Code: "AThrottleableError",
			RetryErrorCodes: []string{
				"RetryMePlease",
				"SomeOtherError",
			},
			ThrottleErrorCodes: []string{
				"AThrottleableError",
				"SomeOtherError",
			},
			Throttle: true,
		},
		"unknown code": {
			Code: "UnknownCode",
			RetryErrorCodes: []string{
				"RetryMePlease",
				"SomeOtherError",
			},
			Retryable: false,
		},
	}

	for name, c := range cases {
		req := Request{
			HTTPRequest:        &http.Request{},
			HTTPResponse:       &http.Response{},
			Error:              awserr.New(c.Code, "some error", nil),
			RetryErrorCodes:    c.RetryErrorCodes,
			ThrottleErrorCodes: c.ThrottleErrorCodes,
		}

		retryable := req.IsErrorRetryable()
		if e, a := c.Retryable, retryable; e != a {
			t.Errorf("%s, expect %v retryable, got %v", name, e, a)
		}

		throttle := req.IsErrorThrottle()
		if e, a := c.Throttle, throttle; e != a {
			t.Errorf("%s, expect %v throttle, got %v", name, e, a)
		}
	}
}
