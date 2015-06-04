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

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/internal/apierr"
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
	var err error
	r.HTTPResponse, err = r.Service.Config.HTTPClient.Do(r.HTTPRequest)
	if err != nil {
		// Capture the case where url.Error is returned for error processing
		// response. e.g. 301 without location header comes back as string
		// error and r.HTTPResponse is nil. Other url redirect errors will
		// comeback in a similar method.
		if e, ok := err.(*url.Error); ok {
			if s := reStatusCode.FindStringSubmatch(e.Error()); s != nil {
				code, _ := strconv.ParseInt(s[1], 10, 64)
				r.HTTPResponse = &http.Response{
					StatusCode: int(code),
					Status:     http.StatusText(int(code)),
					Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
				}
				return
			}
		}
		// Catch all other request errors.
		r.Error = apierr.New("RequestError", "send request failed", err)
		r.Retryable.Set(true) // network errors are retryable
	}
}

// ValidateResponseHandler is a request handler to validate service response.
func ValidateResponseHandler(r *Request) {
	if r.HTTPResponse.StatusCode == 0 || r.HTTPResponse.StatusCode >= 300 {
		// this may be replaced by an UnmarshalError handler
		r.Error = apierr.New("UnknownError", "unknown error", nil)
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
		if r.Error != nil {
			if err, ok := r.Error.(awserr.Error); ok {
				if isCodeExpiredCreds(err.Code()) {
					r.Config.Credentials.Expire()
					// The credentials will need to be resigned with new credentials
					r.signed = false
				}
			}
		}

		r.RetryCount++
		r.Error = nil
	}
}

var (
	// ErrMissingRegion is an error that is returned if region configuration is
	// not found.
	ErrMissingRegion error = apierr.New("MissingRegion", "could not find region configuration", nil)

	// ErrMissingEndpoint is an error that is returned if an endpoint cannot be
	// resolved for a service.
	ErrMissingEndpoint error = apierr.New("MissingEndpoint", "'Endpoint' configuration is required for this service", nil)
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
