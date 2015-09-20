package corehandlers

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"strconv"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// Interface for matching types which also have a Len method.
type lener interface {
	Len() int
}

// BuildContentLength builds the content length of a request based on the body,
// or will use the HTTPRequest.Header's "Content-Length" if defined. If unable
// to determine request body length and no "Content-Length" was specified it will panic.
var BuildContentLengthHandler = request.NamedHandler{"core.BuildContentLengthHandler", func(r *request.Request) {
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
		r.BodyStart, _ = body.Seek(0, 1)
		end, _ := body.Seek(0, 2)
		body.Seek(r.BodyStart, 0) // make sure to seek back to original location
		length = end - r.BodyStart
	default:
		panic("Cannot get length of body, must provide `ContentLength`")
	}

	r.HTTPRequest.ContentLength = length
	r.HTTPRequest.Header.Set("Content-Length", fmt.Sprintf("%d", length))
}}

// UserAgentHandler is a request handler for injecting User agent into requests.
var UserAgentHandler = request.NamedHandler{"core.UserAgentHandler", func(r *request.Request) {
	r.HTTPRequest.Header.Set("User-Agent", aws.SDKName+"/"+aws.SDKVersion)
}}

var reStatusCode = regexp.MustCompile(`^(\d{3})`)

// SendHandler is a request handler to send service request using HTTP client.
var SendHandler = request.NamedHandler{"core.SendHandler", func(r *request.Request) {
	var err error
	r.HTTPResponse, err = r.Service.Config.HTTPClient.Do(r.HTTPRequest)
	if err != nil {
		// Capture the case where url.Error is returned for error processing
		// response. e.g. 301 without location header comes back as string
		// error and r.HTTPResponse is nil. Other url redirect errors will
		// comeback in a similar method.
		if e, ok := err.(*url.Error); ok && e.Err != nil {
			if s := reStatusCode.FindStringSubmatch(e.Err.Error()); s != nil {
				code, _ := strconv.ParseInt(s[1], 10, 64)
				r.HTTPResponse = &http.Response{
					StatusCode: int(code),
					Status:     http.StatusText(int(code)),
					Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
				}
				return
			}
		}
		if r.HTTPResponse == nil {
			// Add a dummy request response object to ensure the HTTPResponse
			// value is consistent.
			r.HTTPResponse = &http.Response{
				StatusCode: int(0),
				Status:     http.StatusText(int(0)),
				Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
			}
		}
		// Catch all other request errors.
		r.Error = awserr.New("RequestError", "send request failed", err)
		r.Retryable = aws.Bool(true) // network errors are retryable
	}
}}

// ValidateResponseHandler is a request handler to validate service response.
var ValidateResponseHandler = request.NamedHandler{"core.ValidateResponseHandler", func(r *request.Request) {
	if r.HTTPResponse.StatusCode == 0 || r.HTTPResponse.StatusCode >= 300 {
		// this may be replaced by an UnmarshalError handler
		r.Error = awserr.New("UnknownError", "unknown error", nil)
	}
}}

// AfterRetryHandler performs final checks to determine if the request should
// be retried and how long to delay.
var AfterRetryHandler = request.NamedHandler{"core.AfterRetryHandler", func(r *request.Request) {
	// If one of the other handlers already set the retry state
	// we don't want to override it based on the service's state
	if r.Retryable == nil {
		r.Retryable = aws.Bool(r.ShouldRetry(r))
	}

	if r.WillRetry() {
		r.RetryDelay = r.RetryRules(r)
		r.Service.Config.SleepDelay(r.RetryDelay)

		// when the expired token exception occurs the credentials
		// need to be expired locally so that the next request to
		// get credentials will trigger a credentials refresh.
		if r.IsErrorExpired() {
			r.Service.Config.Credentials.Expire()
		}

		r.RetryCount++
		r.Error = nil
	}
}}

// ValidateEndpointHandler is a request handler to validate a request had the
// appropriate Region and Endpoint set. Will set r.Error if the endpoint or
// region is not valid.
var ValidateEndpointHandler = request.NamedHandler{"core.ValidateEndpointHandler", func(r *request.Request) {
	if r.Service.SigningRegion == "" && aws.StringValue(r.Service.Config.Region) == "" {
		r.Error = aws.ErrMissingRegion
	} else if r.Service.Endpoint == "" {
		r.Error = aws.ErrMissingEndpoint
	}
}}
