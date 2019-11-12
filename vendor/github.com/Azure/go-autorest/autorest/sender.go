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
	"context"
	"crypto/tls"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/http/cookiejar"
	"strconv"
	"time"

	"github.com/Azure/go-autorest/tracing"
)

// used as a key type in context.WithValue()
type ctxSendDecorators struct{}

// WithSendDecorators adds the specified SendDecorators to the provided context.
// If no SendDecorators are provided the context is unchanged.
func WithSendDecorators(ctx context.Context, sendDecorator []SendDecorator) context.Context {
	if len(sendDecorator) == 0 {
		return ctx
	}
	return context.WithValue(ctx, ctxSendDecorators{}, sendDecorator)
}

// GetSendDecorators returns the SendDecorators in the provided context or the provided default SendDecorators.
func GetSendDecorators(ctx context.Context, defaultSendDecorators ...SendDecorator) []SendDecorator {
	inCtx := ctx.Value(ctxSendDecorators{})
	if sd, ok := inCtx.([]SendDecorator); ok {
		return sd
	}
	return defaultSendDecorators
}

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

// SendDecorator takes and possibly decorates, by wrapping, a Sender. Decorators may affect the
// http.Request and pass it along or, first, pass the http.Request along then react to the
// http.Response result.
type SendDecorator func(Sender) Sender

// CreateSender creates, decorates, and returns, as a Sender, the default http.Client.
func CreateSender(decorators ...SendDecorator) Sender {
	return DecorateSender(sender(tls.RenegotiateNever), decorators...)
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
	return SendWithSender(sender(tls.RenegotiateNever), r, decorators...)
}

// SendWithSender sends the passed http.Request, through the provided Sender, returning the
// http.Response and possible error. It also accepts a, possibly empty, set of SendDecorators which
// it will apply the http.Client before invoking the Do method.
//
// SendWithSender will not poll or retry requests.
func SendWithSender(s Sender, r *http.Request, decorators ...SendDecorator) (*http.Response, error) {
	return DecorateSender(s, decorators...).Do(r)
}

func sender(renengotiation tls.RenegotiationSupport) Sender {
	// Use behaviour compatible with DefaultTransport, but require TLS minimum version.
	defaultTransport := http.DefaultTransport.(*http.Transport)
	transport := &http.Transport{
		Proxy:                 defaultTransport.Proxy,
		DialContext:           defaultTransport.DialContext,
		MaxIdleConns:          defaultTransport.MaxIdleConns,
		IdleConnTimeout:       defaultTransport.IdleConnTimeout,
		TLSHandshakeTimeout:   defaultTransport.TLSHandshakeTimeout,
		ExpectContinueTimeout: defaultTransport.ExpectContinueTimeout,
		TLSClientConfig: &tls.Config{
			MinVersion:    tls.VersionTLS12,
			Renegotiation: renengotiation,
		},
	}
	var roundTripper http.RoundTripper = transport
	if tracing.IsEnabled() {
		roundTripper = tracing.NewTransport(transport)
	}
	j, _ := cookiejar.New(nil)
	return &http.Client{Jar: j, Transport: roundTripper}
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
// time.Duration (which may be zero). Retrying may be canceled by cancelling the context on the http.Request.
// NOTE: Code http.StatusTooManyRequests (429) will *not* be counted against the number of attempts.
func DoRetryForStatusCodes(attempts int, backoff time.Duration, codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			return doRetryForStatusCodesImpl(s, r, false, attempts, backoff, 0, codes...)
		})
	}
}

// DoRetryForStatusCodesWithCap returns a SendDecorator that retries for specified statusCodes for up to the
// specified number of attempts, exponentially backing off between requests using the supplied backoff
// time.Duration (which may be zero). To cap the maximum possible delay between iterations specify a value greater
// than zero for cap. Retrying may be canceled by cancelling the context on the http.Request.
func DoRetryForStatusCodesWithCap(attempts int, backoff, cap time.Duration, codes ...int) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			return doRetryForStatusCodesImpl(s, r, true, attempts, backoff, cap, codes...)
		})
	}
}

func doRetryForStatusCodesImpl(s Sender, r *http.Request, count429 bool, attempts int, backoff, cap time.Duration, codes ...int) (resp *http.Response, err error) {
	rr := NewRetriableRequest(r)
	// Increment to add the first call (attempts denotes number of retries)
	for attempt := 0; attempt < attempts+1; {
		err = rr.Prepare()
		if err != nil {
			return
		}
		resp, err = s.Do(rr.Request())
		// we want to retry if err is not nil (e.g. transient network failure).  note that for failed authentication
		// resp and err will both have a value, so in this case we don't want to retry as it will never succeed.
		if err == nil && !ResponseHasStatusCode(resp, codes...) || IsTokenRefreshError(err) {
			return resp, err
		}
		delayed := DelayWithRetryAfter(resp, r.Context().Done())
		if !delayed && !DelayForBackoffWithCap(backoff, cap, attempt, r.Context().Done()) {
			return resp, r.Context().Err()
		}
		// when count429 == false don't count a 429 against the number
		// of attempts so that we continue to retry until it succeeds
		if count429 || (resp == nil || resp.StatusCode != http.StatusTooManyRequests) {
			attempt++
		}
	}
	return resp, err
}

// DelayWithRetryAfter invokes time.After for the duration specified in the "Retry-After" header.
// The value of Retry-After can be either the number of seconds or a date in RFC1123 format.
// The function returns true after successfully waiting for the specified duration.  If there is
// no Retry-After header or the wait is cancelled the return value is false.
func DelayWithRetryAfter(resp *http.Response, cancel <-chan struct{}) bool {
	if resp == nil {
		return false
	}
	var dur time.Duration
	ra := resp.Header.Get("Retry-After")
	if retryAfter, _ := strconv.Atoi(ra); retryAfter > 0 {
		dur = time.Duration(retryAfter) * time.Second
	} else if t, err := time.Parse(time.RFC1123, ra); err == nil {
		dur = t.Sub(time.Now())
	}
	if dur > 0 {
		select {
		case <-time.After(dur):
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
	return DelayForBackoffWithCap(backoff, 0, attempt, cancel)
}

// DelayForBackoffWithCap invokes time.After for the supplied backoff duration raised to the power of
// passed attempt (i.e., an exponential backoff delay). Backoff duration is in seconds and can set
// to zero for no delay. To cap the maximum possible delay specify a value greater than zero for cap.
// The delay may be canceled by closing the passed channel. If terminated early, returns false.
// Note: Passing attempt 1 will result in doubling "backoff" duration. Treat this as a zero-based attempt
// count.
func DelayForBackoffWithCap(backoff, cap time.Duration, attempt int, cancel <-chan struct{}) bool {
	d := time.Duration(backoff.Seconds()*math.Pow(2, float64(attempt))) * time.Second
	if cap > 0 && d > cap {
		d = cap
	}
	select {
	case <-time.After(d):
		return true
	case <-cancel:
		return false
	}
}
