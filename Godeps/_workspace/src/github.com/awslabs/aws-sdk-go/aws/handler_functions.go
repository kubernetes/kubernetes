package aws

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"time"
)

var sleepDelay = func(delay time.Duration) {
	time.Sleep(delay)
}

// Interface for matching types which also have a Len method.
type lener interface {
	Len() int
}

// BuildContentLength builds the content length of a request based on the body,
// or will use the HTTPRequest.Header's "Content-Length" if defined. If unable
// to determine request body length and no "Content-Length" was specified it will panic.
func BuildContentLength(r *Request) {
	if slength := r.HTTPRequest.Header.Get("Content-Length"); slength != "" {
		length, _ := strconv.ParseInt(slength, 10, 64)
		r.HTTPRequest.ContentLength = length
		return
	}

	var length int64
	switch body := r.Body.(type) {
	case nil:
		length = 0
	case lener:
		length = int64(body.Len())
	case io.Seeker:
		r.bodyStart, _ = body.Seek(0, 1)
		end, _ := body.Seek(0, 2)
		body.Seek(r.bodyStart, 0) // make sure to seek back to original location
		length = end - r.bodyStart
	default:
		panic("Cannot get length of body, must provide `ContentLength`")
	}

	r.HTTPRequest.ContentLength = length
	r.HTTPRequest.Header.Set("Content-Length", fmt.Sprintf("%d", length))
}

// UserAgentHandler is a request handler for injecting User agent into requests.
func UserAgentHandler(r *Request) {
	r.HTTPRequest.Header.Set("User-Agent", SDKName+"/"+SDKVersion)
}

var reStatusCode = regexp.MustCompile(`^(\d+)`)

// SendHandler is a request handler to send service request using HTTP client.
func SendHandler(r *Request) {
	r.HTTPResponse, r.Error = r.Service.Config.HTTPClient.Do(r.HTTPRequest)
	if r.Error != nil {
		if e, ok := r.Error.(*url.Error); ok {
			if s := reStatusCode.FindStringSubmatch(e.Err.Error()); s != nil {
				code, _ := strconv.ParseInt(s[1], 10, 64)
				r.Error = nil
				r.HTTPResponse = &http.Response{
					StatusCode: int(code),
					Status:     http.StatusText(int(code)),
					Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
				}
			}
		}
	}
}

// ValidateResponseHandler is a request handler to validate service response.
func ValidateResponseHandler(r *Request) {
	if r.HTTPResponse.StatusCode == 0 || r.HTTPResponse.StatusCode >= 300 {
		// this may be replaced by an UnmarshalError handler
		r.Error = &APIError{
			StatusCode: r.HTTPResponse.StatusCode,
			Code:       "UnknownError",
			Message:    "unknown error",
		}
	}
}

// AfterRetryHandler performs final checks to determine if the request should
// be retried and how long to delay.
func AfterRetryHandler(r *Request) {
	// If one of the other handlers already set the retry state
	// we don't want to override it based on the service's state
	if !r.Retryable.IsSet() {
		r.Retryable.Set(r.Service.ShouldRetry(r))
	}

	if r.WillRetry() {
		r.RetryDelay = r.Service.RetryRules(r)
		sleepDelay(r.RetryDelay)

		// when the expired token exception occurs the credentials
		// need to be expired locally so that the next request to
		// get credentials will trigger a credentials refresh.
		if err := Error(r.Error); err != nil && err.Code == "ExpiredTokenException" {
			r.Config.Credentials.Expire()
			// The credentials will need to be resigned with new credentials
			r.signed = false
		}

		r.RetryCount++
		r.Error = nil
	}
}

var (
	// ErrMissingRegion is an error that is returned if region configuration is
	// not found.
	ErrMissingRegion = fmt.Errorf("could not find region configuration")

	// ErrMissingEndpoint is an error that is returned if an endpoint cannot be
	// resolved for a service.
	ErrMissingEndpoint = fmt.Errorf("`Endpoint' configuration is required for this service")
)

// ValidateEndpointHandler is a request handler to validate a request had the
// appropriate Region and Endpoint set. Will set r.Error if the endpoint or
// region is not valid.
func ValidateEndpointHandler(r *Request) {
	if r.Service.SigningRegion == "" && r.Service.Config.Region == "" {
		r.Error = ErrMissingRegion
	} else if r.Service.Endpoint == "" {
		r.Error = ErrMissingEndpoint
	}
}
