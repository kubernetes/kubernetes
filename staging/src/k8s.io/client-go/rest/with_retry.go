/*
Copyright 2021 The Kubernetes Authors.

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

package rest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"k8s.io/klog/v2"
)

// IsRetryableErrorFunc allows the client to provide its own function
// that determines whether the specified err from the server is retryable.
//
// request: the original request sent to the server
// err: the server sent this error to us
//
// The function returns true if the error is retryable and the request
// can be retried, otherwise it returns false.
// We have four mode of communications - 'Stream', 'Watch', 'Do' and 'DoRaw', this
// function allows us to customize the retryability aspect of each.
type IsRetryableErrorFunc func(request *http.Request, err error) bool

func (r IsRetryableErrorFunc) IsErrorRetryable(request *http.Request, err error) bool {
	return r(request, err)
}

var neverRetryError = IsRetryableErrorFunc(func(_ *http.Request, _ error) bool {
	return false
})

// WithRetry allows the client to retry a request up to a certain number of times
// Note that WithRetry is not safe for concurrent use by multiple
// goroutines without additional locking or coordination.
type WithRetry interface {
	// IsNextRetry advances the retry counter appropriately
	// and returns true if the request should be retried,
	// otherwise it returns false, if:
	//  - we have already reached the maximum retry threshold.
	//  - the error does not fall into the retryable category.
	//  - the server has not sent us a 429, or 5xx status code and the
	//    'Retry-After' response header is not set with a value.
	//  - we need to seek to the beginning of the request body before we
	//    initiate the next retry, the function should log an error and
	//    return false if it fails to do so.
	//
	// restReq: the associated rest.Request
	// httpReq: the HTTP Request sent to the server
	// resp: the response sent from the server, it is set if err is nil
	// err: the server sent this error to us, if err is set then resp is nil.
	// f: a IsRetryableErrorFunc function provided by the client that determines
	//    if the err sent by the server is retryable.
	IsNextRetry(ctx context.Context, restReq *Request, httpReq *http.Request, resp *http.Response, err error, f IsRetryableErrorFunc) bool

	// Before should be invoked prior to each attempt, including
	// the first one. If an error is returned, the request should
	// be aborted immediately.
	//
	// Before may also be additionally responsible for preparing
	// the request for the next retry, namely in terms of resetting
	// the request body in case it has been read.
	Before(ctx context.Context, r *Request) error

	// After should be invoked immediately after an attempt is made.
	After(ctx context.Context, r *Request, resp *http.Response, err error)

	// WrapPreviousError wraps the error from any previous attempt into
	// the final error specified in 'finalErr', so the user has more
	// context why the request failed.
	// For example, if a request times out after multiple retries then
	// we see a generic context.Canceled or context.DeadlineExceeded
	// error which is not very useful in debugging. This function can
	// wrap any error from previous attempt(s) to provide more context to
	// the user. The error returned in 'err' must satisfy the
	// following conditions:
	//  a: errors.Unwrap(err) = errors.Unwrap(finalErr) if finalErr
	//     implements Unwrap
	//  b: errors.Unwrap(err) = finalErr if finalErr does not
	//     implements Unwrap
	//  c: errors.Is(err, otherErr) = errors.Is(finalErr, otherErr)
	WrapPreviousError(finalErr error) (err error)
}

// RetryAfter holds information associated with the next retry.
type RetryAfter struct {
	// Wait is the duration the server has asked us to wait before
	// the next retry is initiated.
	// This is the value of the 'Retry-After' response header in seconds.
	Wait time.Duration

	// Attempt is the Nth attempt after which we have received a retryable
	// error or a 'Retry-After' response header from the server.
	Attempt int

	// Reason describes why we are retrying the request
	Reason string
}

type withRetry struct {
	maxRetries int
	attempts   int

	// retry after parameters that pertain to the attempt that is to
	// be made soon, so as to enable 'Before' and 'After' to refer
	// to the retry parameters.
	//  - for the first attempt, it will always be nil
	//  - for consecutive attempts, it is non nil and holds the
	//    retry after parameters for the next attempt to be made.
	retryAfter *RetryAfter

	// we keep track of two most recent errors, if the most
	// recent attempt is labeled as 'N' then:
	//  - currentErr represents the error returned by attempt N, it
	//    can be nil if attempt N did not return an error.
	//  - previousErr represents an error from an attempt 'M' which
	//    precedes attempt 'N' (N - M >= 1), it is non nil only when:
	//      - for a sequence of attempt(s) 1..n (n>1), there
	//        is an attempt k (k<n) that returned an error.
	previousErr, currentErr error
}

func (r *withRetry) trackPreviousError(err error) {
	// keep track of two most recent errors
	if r.currentErr != nil {
		r.previousErr = r.currentErr
	}
	r.currentErr = err
}

func (r *withRetry) IsNextRetry(ctx context.Context, restReq *Request, httpReq *http.Request, resp *http.Response, err error, f IsRetryableErrorFunc) bool {
	defer r.trackPreviousError(err)

	if httpReq == nil || (resp == nil && err == nil) {
		// bad input, we do nothing.
		return false
	}

	if restReq.body != nil {
		// we have an opaque reader, we can't safely reset it
		return false
	}

	r.attempts++
	r.retryAfter = &RetryAfter{Attempt: r.attempts}
	if r.attempts > r.maxRetries {
		return false
	}

	// if the server returned an error, it takes precedence over the http response.
	var errIsRetryable bool
	if f != nil && err != nil && f.IsErrorRetryable(httpReq, err) {
		errIsRetryable = true
		// we have a retryable error, for which we will create an
		// artificial "Retry-After" response.
		resp = retryAfterResponse()
	}
	if err != nil && !errIsRetryable {
		return false
	}

	// if we are here, we have either a or b:
	//  a: we have a retryable error, for which we already
	//     have an artificial "Retry-After" response.
	//  b: we have a response from the server for which we
	//     need to check if it is retryable
	seconds, wait := checkWait(resp)
	if !wait {
		return false
	}

	r.retryAfter.Wait = time.Duration(seconds) * time.Second
	r.retryAfter.Reason = getRetryReason(r.attempts, seconds, resp, err)

	return true
}

func (r *withRetry) Before(ctx context.Context, request *Request) error {
	// If the request context is already canceled there
	// is no need to retry.
	if ctx.Err() != nil {
		r.trackPreviousError(ctx.Err())
		return ctx.Err()
	}

	url := request.URL()
	// r.retryAfter represents the retry after parameters calculated
	// from the (response, err) tuple from the last attempt, so 'Before'
	// can apply these retry after parameters prior to the next attempt.
	// 'r.retryAfter == nil' indicates that this is the very first attempt.
	if r.retryAfter == nil {
		// we do a backoff sleep before the first attempt is made,
		// (preserving current behavior).
		if request.backoff != nil {
			request.backoff.Sleep(request.backoff.CalculateBackoff(url))
		}
		return nil
	}

	// if we are here, we have made attempt(s) at least once before.
	if request.backoff != nil {
		delay := request.backoff.CalculateBackoff(url)
		if r.retryAfter.Wait > delay {
			delay = r.retryAfter.Wait
		}
		request.backoff.Sleep(delay)
	}

	// We are retrying the request that we already send to
	// apiserver at least once before. This request should
	// also be throttled with the client-internal rate limiter.
	if err := request.tryThrottleWithInfo(ctx, r.retryAfter.Reason); err != nil {
		r.trackPreviousError(ctx.Err())
		return err
	}

	klog.V(4).Infof("Got a Retry-After %s response for attempt %d to %v", r.retryAfter.Wait, r.retryAfter.Attempt, request.URL().String())
	return nil
}

func (r *withRetry) After(ctx context.Context, request *Request, resp *http.Response, err error) {
	// 'After' is invoked immediately after an attempt is made, let's label
	// the attempt we have just made as attempt 'N'.
	// the current value of r.retryAfter represents the retry after
	// parameters calculated from the (response, err) tuple from
	// attempt N-1, so r.retryAfter is outdated and should not be
	// referred to here.
	r.retryAfter = nil

	if request.c.base != nil {
		if err != nil {
			request.backoff.UpdateBackoff(request.URL(), err, 0)
		} else {
			request.backoff.UpdateBackoff(request.URL(), err, resp.StatusCode)
		}
	}
}

func (r *withRetry) WrapPreviousError(currentErr error) error {
	if currentErr == nil || r.previousErr == nil {
		return currentErr
	}

	// if both previous and current error objects represent the error,
	// then there is no need to wrap the previous error.
	if currentErr.Error() == r.previousErr.Error() {
		return currentErr
	}

	previousErr := r.previousErr
	// net/http wraps the underlying error with an url.Error, if the
	// previous err object is an instance of url.Error, then we can
	// unwrap it to get to the inner error object, this is so we can
	// avoid error message like:
	//  Error: Get "http://foo.bar/api/v1": context deadline exceeded - error \
	//  from a previous attempt: Error: Get "http://foo.bar/api/v1": EOF
	if urlErr, ok := r.previousErr.(*url.Error); ok && urlErr != nil {
		if urlErr.Unwrap() != nil {
			previousErr = urlErr.Unwrap()
		}
	}

	return &wrapPreviousError{
		currentErr:    currentErr,
		previousError: previousErr,
	}
}

type wrapPreviousError struct {
	currentErr, previousError error
}

func (w *wrapPreviousError) Unwrap() error { return w.currentErr }
func (w *wrapPreviousError) Error() string {
	return fmt.Sprintf("%s - error from a previous attempt: %s", w.currentErr.Error(), w.previousError.Error())
}

// checkWait returns true along with a number of seconds if
// the server instructed us to wait before retrying.
func checkWait(resp *http.Response) (int, bool) {
	switch r := resp.StatusCode; {
	// any 500 error code and 429 can trigger a wait
	case r == http.StatusTooManyRequests, r >= 500:
	default:
		return 0, false
	}
	i, ok := retryAfterSeconds(resp)
	return i, ok
}

func getRetryReason(retries, seconds int, resp *http.Response, err error) string {
	// priority and fairness sets the UID of the FlowSchema
	// associated with a request in the following response Header.
	const responseHeaderMatchedFlowSchemaUID = "X-Kubernetes-PF-FlowSchema-UID"

	message := fmt.Sprintf("retries: %d, retry-after: %ds", retries, seconds)

	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		// it is server-side throttling from priority and fairness
		flowSchemaUID := resp.Header.Get(responseHeaderMatchedFlowSchemaUID)
		return fmt.Sprintf("%s - retry-reason: due to server-side throttling, FlowSchema UID: %q", message, flowSchemaUID)
	case err != nil:
		// it's a retryable error
		return fmt.Sprintf("%s - retry-reason: due to retryable error, error: %v", message, err)
	default:
		return fmt.Sprintf("%s - retry-reason: %d", message, resp.StatusCode)
	}
}

func readAndCloseResponseBody(resp *http.Response) {
	if resp == nil {
		return
	}

	// Ensure the response body is fully read and closed
	// before we reconnect, so that we reuse the same TCP
	// connection.
	const maxBodySlurpSize = 2 << 10
	defer resp.Body.Close()

	if resp.ContentLength <= maxBodySlurpSize {
		io.Copy(io.Discard, &io.LimitedReader{R: resp.Body, N: maxBodySlurpSize})
	}
}

func retryAfterResponse() *http.Response {
	return retryAfterResponseWithDelay("1")
}

func retryAfterResponseWithDelay(delay string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusInternalServerError,
		Header:     http.Header{"Retry-After": []string{delay}},
	}
}
