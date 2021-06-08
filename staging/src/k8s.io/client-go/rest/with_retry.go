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
	"io/ioutil"
	"net/http"
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
	// SetMaxRetries makes the request use the specified integer as a ceiling
	// for retries upon receiving a 429 status code  and the "Retry-After" header
	// in the response.
	// A zero maxRetries should prevent from doing any retry and return immediately.
	SetMaxRetries(maxRetries int)

	// NextRetry advances the retry counter appropriately and returns true if the
	// request should be retried, otherwise it returns false if:
	//  - we have already reached the maximum retry threshold.
	//  - the error does not fall into the retryable category.
	//  - the server has not sent us a 429, or 5xx status code and the
	//    'Retry-After' response header is not set with a value.
	//
	// if retry is set to true, retryAfter will contain the information
	// regarding the next retry.
	//
	// request: the original request sent to the server
	// resp: the response sent from the server, it is set if err is nil
	// err: the server sent this error to us, if err is set then resp is nil.
	// f: a IsRetryableErrorFunc function provided by the client that determines
	//    if the err sent by the server is retryable.
	NextRetry(req *http.Request, resp *http.Response, err error, f IsRetryableErrorFunc) (*RetryAfter, bool)

	// BeforeNextRetry is responsible for carrying out operations that need
	// to be completed before the next retry is initiated:
	// - if the request context is already canceled there is no need to
	//   retry, the function will return ctx.Err().
	// - we need to seek to the beginning of the request body before we
	//   initiate the next retry, the function should return an error if
	//   it fails to do so.
	// - we should wait the number of seconds the server has asked us to
	//   in the 'Retry-After' response header.
	//
	// If BeforeNextRetry returns an error the client should abort the retry,
	// otherwise it is safe to initiate the next retry.
	BeforeNextRetry(ctx context.Context, backoff BackoffManager, retryAfter *RetryAfter, url string, body io.Reader) error
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
}

func (r *withRetry) SetMaxRetries(maxRetries int) {
	if maxRetries < 0 {
		maxRetries = 0
	}
	r.maxRetries = maxRetries
}

func (r *withRetry) NextRetry(req *http.Request, resp *http.Response, err error, f IsRetryableErrorFunc) (*RetryAfter, bool) {
	if req == nil || (resp == nil && err == nil) {
		// bad input, we do nothing.
		return nil, false
	}

	r.attempts++
	retryAfter := &RetryAfter{Attempt: r.attempts}
	if r.attempts > r.maxRetries {
		return retryAfter, false
	}

	// if the server returned an error, it takes precedence over the http response.
	var errIsRetryable bool
	if f != nil && err != nil && f.IsErrorRetryable(req, err) {
		errIsRetryable = true
		// we have a retryable error, for which we will create an
		// artificial "Retry-After" response.
		resp = retryAfterResponse()
	}
	if err != nil && !errIsRetryable {
		return retryAfter, false
	}

	// if we are here, we have either a or b:
	//  a: we have a retryable error, for which we already
	//     have an artificial "Retry-After" response.
	//  b: we have a response from the server for which we
	//     need to check if it is retryable
	seconds, wait := checkWait(resp)
	if !wait {
		return retryAfter, false
	}

	retryAfter.Wait = time.Duration(seconds) * time.Second
	retryAfter.Reason = getRetryReason(r.attempts, seconds, resp, err)
	return retryAfter, true
}

func (r *withRetry) BeforeNextRetry(ctx context.Context, backoff BackoffManager, retryAfter *RetryAfter, url string, body io.Reader) error {
	// Ensure the response body is fully read and closed before
	// we reconnect, so that we reuse the same TCP connection.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	if seeker, ok := body.(io.Seeker); ok && body != nil {
		if _, err := seeker.Seek(0, 0); err != nil {
			return fmt.Errorf("can't Seek() back to beginning of body for %T", r)
		}
	}

	klog.V(4).Infof("Got a Retry-After %s response for attempt %d to %v", retryAfter.Wait, retryAfter.Attempt, url)
	if backoff != nil {
		backoff.Sleep(retryAfter.Wait)
	}
	return nil
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
		io.Copy(ioutil.Discard, &io.LimitedReader{R: resp.Body, N: maxBodySlurpSize})
	}
}

func retryAfterResponse() *http.Response {
	return &http.Response{
		StatusCode: http.StatusInternalServerError,
		Header:     http.Header{"Retry-After": []string{"1"}},
	}
}
