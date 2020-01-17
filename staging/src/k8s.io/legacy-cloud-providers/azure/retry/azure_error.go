// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package retry

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog"
)

const (
	// RetryAfterHeaderKey is the retry-after header key in ARM responses.
	RetryAfterHeaderKey = "Retry-After"
)

var (
	// The function to get current time.
	now = time.Now
)

// Error indicates an error returned by Azure APIs.
type Error struct {
	// Retriable indicates whether the request is retriable.
	Retriable bool
	// HTTPStatusCode indicates the response HTTP status code.
	HTTPStatusCode int
	// RetryAfter indicates the time when the request should retry after throttling.
	// A throttled request is retriable.
	RetryAfter time.Time
	// RetryAfter indicates the raw error from API.
	RawError error
}

// Error returns the error.
// Note that Error doesn't implement error interface because (nil *Error) != (nil error).
func (err *Error) Error() error {
	if err == nil {
		return nil
	}

	// Convert time to seconds for better logging.
	retryAfterSeconds := 0
	curTime := now()
	if err.RetryAfter.After(curTime) {
		retryAfterSeconds = int(err.RetryAfter.Sub(curTime) / time.Second)
	}

	return fmt.Errorf("Retriable: %v, RetryAfter: %ds, HTTPStatusCode: %d, RawError: %v",
		err.Retriable, retryAfterSeconds, err.HTTPStatusCode, err.RawError)
}

// IsThrottled returns true the if the request is being throttled.
func (err *Error) IsThrottled() bool {
	if err == nil {
		return false
	}

	return err.HTTPStatusCode == http.StatusTooManyRequests || err.RetryAfter.After(now())
}

// NewError creates a new Error.
func NewError(retriable bool, err error) *Error {
	return &Error{
		Retriable: retriable,
		RawError:  err,
	}
}

// GetRetriableError gets new retriable Error.
func GetRetriableError(err error) *Error {
	return &Error{
		Retriable: true,
		RawError:  err,
	}
}

// GetRateLimitError creates a new error for rate limiting.
func GetRateLimitError(isWrite bool, opName string) *Error {
	opType := "read"
	if isWrite {
		opType = "write"
	}
	return GetRetriableError(fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", opType, opName))
}

// GetThrottlingError creates a new error for throttling.
func GetThrottlingError(operation, reason string, retryAfter time.Time) *Error {
	rawError := fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", operation, reason)
	return &Error{
		Retriable:  true,
		RawError:   rawError,
		RetryAfter: retryAfter,
	}
}

// GetError gets a new Error based on resp and error.
func GetError(resp *http.Response, err error) *Error {
	if err == nil && resp == nil {
		return nil
	}

	if err == nil && resp != nil && isSuccessHTTPResponse(resp) {
		// HTTP 2xx suggests a successful response
		return nil
	}

	retryAfter := time.Time{}
	if retryAfterDuration := getRetryAfter(resp); retryAfterDuration != 0 {
		retryAfter = now().Add(retryAfterDuration)
	}
	return &Error{
		RawError:       getRawError(resp, err),
		RetryAfter:     retryAfter,
		Retriable:      shouldRetryHTTPRequest(resp, err),
		HTTPStatusCode: getHTTPStatusCode(resp),
	}
}

// isSuccessHTTPResponse determines if the response from an HTTP request suggests success
func isSuccessHTTPResponse(resp *http.Response) bool {
	if resp == nil {
		return false
	}

	// HTTP 2xx suggests a successful response
	if 199 < resp.StatusCode && resp.StatusCode < 300 {
		return true
	}

	return false
}

func getRawError(resp *http.Response, err error) error {
	if err != nil {
		return err
	}

	if resp == nil || resp.Body == nil {
		return fmt.Errorf("empty HTTP response")
	}

	// return the http status if unabled to get response body.
	defer resp.Body.Close()
	respBody, _ := ioutil.ReadAll(resp.Body)
	resp.Body = ioutil.NopCloser(bytes.NewReader(respBody))
	if len(respBody) == 0 {
		return fmt.Errorf("HTTP status code (%d)", resp.StatusCode)
	}

	// return the raw response body.
	return fmt.Errorf("%s", string(respBody))
}

func getHTTPStatusCode(resp *http.Response) int {
	if resp == nil {
		return -1
	}

	return resp.StatusCode
}

// shouldRetryHTTPRequest determines if the request is retriable.
func shouldRetryHTTPRequest(resp *http.Response, err error) bool {
	if resp != nil {
		// HTTP 412 (StatusPreconditionFailed) means etag mismatch
		// HTTP 400 (BadRequest) means the request cannot be accepted, hence we shouldn't retry.
		if resp.StatusCode == http.StatusPreconditionFailed || resp.StatusCode == http.StatusBadRequest {
			return false
		}

		// HTTP 4xx (except 412) or 5xx suggests we should retry.
		if 399 < resp.StatusCode && resp.StatusCode < 600 {
			return true
		}
	}

	if err != nil {
		return true
	}

	return false
}

// getRetryAfter gets the retryAfter from http response.
// The value of Retry-After can be either the number of seconds or a date in RFC1123 format.
func getRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}

	ra := resp.Header.Get(RetryAfterHeaderKey)
	if ra == "" {
		return 0
	}

	var dur time.Duration
	if retryAfter, _ := strconv.Atoi(ra); retryAfter > 0 {
		dur = time.Duration(retryAfter) * time.Second
	} else if t, err := time.Parse(time.RFC1123, ra); err == nil {
		dur = t.Sub(now())
	}
	return dur
}

// GetStatusNotFoundAndForbiddenIgnoredError gets an error with StatusNotFound and StatusForbidden ignored.
// It is only used in DELETE operations.
func GetStatusNotFoundAndForbiddenIgnoredError(resp *http.Response, err error) *Error {
	rerr := GetError(resp, err)
	if rerr == nil {
		return nil
	}

	// Returns nil when it is StatusNotFound error.
	if rerr.HTTPStatusCode == http.StatusNotFound {
		klog.V(3).Infof("Ignoring StatusNotFound error: %v", rerr)
		return nil
	}

	// Returns nil if the status code is StatusForbidden.
	// This happens when AuthorizationFailed is reported from Azure API.
	if rerr.HTTPStatusCode == http.StatusForbidden {
		klog.V(3).Infof("Ignoring StatusForbidden error: %v", rerr)
		return nil
	}

	return rerr
}

// IsErrorRetriable returns true if the error is retriable.
func IsErrorRetriable(err error) bool {
	if err == nil {
		return false
	}

	return strings.Contains(err.Error(), "Retriable: true")
}
