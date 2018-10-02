package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"time"
)

// Sender is the interface that wraps the Do method to send HTTP requests.
//
// The standard http.Client conforms to this interface.
type Sender interface {
	Do(*http.Request) (*http.Response, error)
}

// SenderFunc is a method that implements the Sender interface.
type SenderFunc func(*http.Request) (*http.Response, error)

// Do implements the Sender interface on SenderFunc.
func (sf SenderFunc) Do(r *http.Request) (*http.Response, error) {
	return sf(r)
}

// SendDecorator takes and possibily decorates, by wrapping, a Sender. Decorators may affect the
// http.Request and pass it along or, first, pass the http.Request along then react to the
// http.Response result.
type SendDecorator func(Sender) Sender

// CreateSender creates, decorates, and returns, as a Sender, the default http.Client.
func CreateSender(decorators ...SendDecorator) Sender {
	return DecorateSender(&http.Client{}, decorators...)
}

// DecorateSender accepts a Sender and a, possibly empty, set of SendDecorators, which is applies to
// the Sender. Decorators are applied in the order received, but their affect upon the request
// depends on whether they are a pre-decorator (change the http.Request and then pass it along) or a
// post-decorator (pass the http.Request along and react to the results in http.Response).
func DecorateSender(s Sender, decorators ...SendDecorator) Sender {
	for _, decorate := range decorators {
		s = decorate(s)
	}
	return s
}

// Send sends, by means of the default http.Client, the passed http.Request, returning the
// http.Response and possible error. It also accepts a, possibly empty, set of SendDecorators which
// it will apply the http.Client before invoking the Do method.
//
// Send is a convenience method and not recommended for production. Advanced users should use
// SendWithSender, passing and sharing their own Sender (e.g., instance of http.Client).
//
// Send will not poll or retry requests.
func Send(r *http.Request, decorators ...SendDecorator) (*http.Response, error) {
	return SendWithSender(&http.Client{}, r, decorators...)
}

// SendWithSender sends the passed http.Request, through the provided Sender, returning the
// http.Response and possible error. It also accepts a, possibly empty, set of SendDecorators which
// it will apply the http.Client before invoking the Do method.
//
// SendWithSender will not poll or retry requests.
func SendWithSender(s Sender, r *http.Request, decorators ...SendDecorator) (*http.Response, error) {
	return DecorateSender(s, decorators...).Do(r)
}

// AfterDelay returns a SendDecorator that delays for the passed time.Duration before
// invoking the Sender. The delay may be terminated by closing the optional channel on the
// http.Request. If canceled, no further Senders are invoked.
func AfterDelay(d time.Duration) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			if !DelayForBackoff(d, 0, r.Context().Done()) {
				return nil, fmt.Errorf("autorest: AfterDelay canceled before full delay")
			}
			return s.Do(r)
		})
	}
}

// AsIs returns a SendDecorator that invokes the passed Sender without modifying the http.Request.
func AsIs() SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			return s.Do(r)
		})
	}
}

// DoCloseIfError returns a SendDecorator that first invokes the passed Sender after which
// it closes the response if the passed Sender returns an error and the response body exists.
func DoCloseIfError() SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err != nil {
				Respond(resp, ByDiscardingBody(), ByClosing())
			}
			return resp, err
		})
	}
}

// DoErrorIfStatusCode returns a SendDecorator that emits an error if the response StatusCode is
// among the set passed. Since these are artificial errors, the response body may still require
// closing.
func DoErrorIfStatusCode(codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err == nil && ResponseHasStatusCode(resp, codes...) {
				err = NewErrorWithResponse("autorest", "DoErrorIfStatusCode", resp, "%v %v failed with %s",
					resp.Request.Method,
					resp.Request.URL,
					resp.Status)
			}
			return resp, err
		})
	}
}

// DoErrorUnlessStatusCode returns a SendDecorator that emits an error unless the response
// StatusCode is among the set passed. Since these are artificial errors, the response body
// may still require closing.
func DoErrorUnlessStatusCode(codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err == nil && !ResponseHasStatusCode(resp, codes...) {
				err = NewErrorWithResponse("autorest", "DoErrorUnlessStatusCode", resp, "%v %v failed with %s",
					resp.Request.Method,
					resp.Request.URL,
					resp.Status)
			}
			return resp, err
		})
	}
}

// DoPollForStatusCodes returns a SendDecorator that polls if the http.Response contains one of the
// passed status codes. It expects the http.Response to contain a Location header providing the
// URL at which to poll (using GET) and will poll until the time passed is equal to or greater than
// the supplied duration. It will delay between requests for the duration specified in the
// RetryAfter header or, if the header is absent, the passed delay. Polling may be canceled by
// closing the optional channel on the http.Request.
func DoPollForStatusCodes(duration time.Duration, delay time.Duration, codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			resp, err = s.Do(r)

			if err == nil && ResponseHasStatusCode(resp, codes...) {
				r, err = NewPollingRequestWithContext(r.Context(), resp)

				for err == nil && ResponseHasStatusCode(resp, codes...) {
					Respond(resp,
						ByDiscardingBody(),
						ByClosing())
					resp, err = SendWithSender(s, r,
						AfterDelay(GetRetryAfter(resp, delay)))
				}
			}

			return resp, err
		})
	}
}

// DoRetryForAttempts returns a SendDecorator that retries a failed request for up to the specified
// number of attempts, exponentially backing off between requests using the supplied backoff
// time.Duration (which may be zero). Retrying may be canceled by closing the optional channel on
// the http.Request.
func DoRetryForAttempts(attempts int, backoff time.Duration) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			rr := NewRetriableRequest(r)
			for attempt := 0; attempt < attempts; attempt++ {
				err = rr.Prepare()
				if err != nil {
					return resp, err
				}
				resp, err = s.Do(rr.Request())
				if err == nil {
					return resp, err
				}
				if !DelayForBackoff(backoff, attempt, r.Context().Done()) {
					return nil, r.Context().Err()
				}
			}
			return resp, err
		})
	}
}

// DoRetryForStatusCodes returns a SendDecorator that retries for specified statusCodes for up to the specified
// number of attempts, exponentially backing off between requests using the supplied backoff
// time.Duration (which may be zero). Retrying may be canceled by closing the optional channel on
// the http.Request.
func DoRetryForStatusCodes(attempts int, backoff time.Duration, codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			rr := NewRetriableRequest(r)
			// Increment to add the first call (attempts denotes number of retries)
			attempts++
			for attempt := 0; attempt < attempts; {
				err = rr.Prepare()
				if err != nil {
					return resp, err
				}
				resp, err = s.Do(rr.Request())
				// if the error isn't temporary don't bother retrying
				if err != nil && !IsTemporaryNetworkError(err) {
					return nil, err
				}
				// we want to retry if err is not nil (e.g. transient network failure).  note that for failed authentication
				// resp and err will both have a value, so in this case we don't want to retry as it will never succeed.
				if err == nil && !ResponseHasStatusCode(resp, codes...) || IsTokenRefreshError(err) {
					return resp, err
				}
				delayed := DelayWithRetryAfter(resp, r.Context().Done())
				if !delayed && !DelayForBackoff(backoff, attempt, r.Context().Done()) {
					return nil, r.Context().Err()
				}
				// don't count a 429 against the number of attempts
				// so that we continue to retry until it succeeds
				if resp == nil || resp.StatusCode != http.StatusTooManyRequests {
					attempt++
				}
			}
			return resp, err
		})
	}
}

// DelayWithRetryAfter invokes time.After for the duration specified in the "Retry-After" header in
// responses with status code 429
func DelayWithRetryAfter(resp *http.Response, cancel <-chan struct{}) bool {
	if resp == nil {
		return false
	}
	retryAfter, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
	if resp.StatusCode == http.StatusTooManyRequests && retryAfter > 0 {
		select {
		case <-time.After(time.Duration(retryAfter) * time.Second):
			return true
		case <-cancel:
			return false
		}
	}
	return false
}

// DoRetryForDuration returns a SendDecorator that retries the request until the total time is equal
// to or greater than the specified duration, exponentially backing off between requests using the
// supplied backoff time.Duration (which may be zero). Retrying may be canceled by closing the
// optional channel on the http.Request.
func DoRetryForDuration(d time.Duration, backoff time.Duration) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			rr := NewRetriableRequest(r)
			end := time.Now().Add(d)
			for attempt := 0; time.Now().Before(end); attempt++ {
				err = rr.Prepare()
				if err != nil {
					return resp, err
				}
				resp, err = s.Do(rr.Request())
				if err == nil {
					return resp, err
				}
				if !DelayForBackoff(backoff, attempt, r.Context().Done()) {
					return nil, r.Context().Err()
				}
			}
			return resp, err
		})
	}
}

// WithLogging returns a SendDecorator that implements simple before and after logging of the
// request.
func WithLogging(logger *log.Logger) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			logger.Printf("Sending %s %s", r.Method, r.URL)
			resp, err := s.Do(r)
			if err != nil {
				logger.Printf("%s %s received error '%v'", r.Method, r.URL, err)
			} else {
				logger.Printf("%s %s received %s", r.Method, r.URL, resp.Status)
			}
			return resp, err
		})
	}
}

// DelayForBackoff invokes time.After for the supplied backoff duration raised to the power of
// passed attempt (i.e., an exponential backoff delay). Backoff duration is in seconds and can set
// to zero for no delay. The delay may be canceled by closing the passed channel. If terminated early,
// returns false.
// Note: Passing attempt 1 will result in doubling "backoff" duration. Treat this as a zero-based attempt
// count.
func DelayForBackoff(backoff time.Duration, attempt int, cancel <-chan struct{}) bool {
	select {
	case <-time.After(time.Duration(backoff.Seconds()*math.Pow(2, float64(attempt))) * time.Second):
		return true
	case <-cancel:
		return false
	}
}
