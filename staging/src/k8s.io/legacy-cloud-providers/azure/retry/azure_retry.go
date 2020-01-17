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
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
	"k8s.io/klog"
)

// Ensure package autorest/mocks is imported and vendored.
var _ autorest.Sender = mocks.NewSender()

// Backoff holds parameters applied to a Backoff function.
type Backoff struct {
	// The initial duration.
	Duration time.Duration
	// Duration is multiplied by factor each iteration, if factor is not zero
	// and the limits imposed by Steps and Cap have not been reached.
	// Should not be negative.
	// The jitter does not contribute to the updates to the duration parameter.
	Factor float64
	// The sleep at each iteration is the duration plus an additional
	// amount chosen uniformly at random from the interval between
	// zero and `jitter*duration`.
	Jitter float64
	// The remaining number of iterations in which the duration
	// parameter may change (but progress can be stopped earlier by
	// hitting the cap). If not positive, the duration is not
	// changed. Used for exponential backoff in combination with
	// Factor and Cap.
	Steps int
	// A limit on revised values of the duration parameter. If a
	// multiplication by the factor parameter would make the duration
	// exceed the cap then the duration is set to the cap and the
	// steps parameter is set to zero.
	Cap time.Duration
	// The errors indicate that the request shouldn't do more retrying.
	NonRetriableErrors []string
}

// NewBackoff creates a new Backoff.
func NewBackoff(duration time.Duration, factor float64, jitter float64, steps int, cap time.Duration) *Backoff {
	return &Backoff{
		Duration: duration,
		Factor:   factor,
		Jitter:   jitter,
		Steps:    steps,
		Cap:      cap,
	}
}

// WithNonRetriableErrors returns a new *Backoff with NonRetriableErrors assigned.
func (b *Backoff) WithNonRetriableErrors(errs []string) *Backoff {
	newBackoff := *b
	newBackoff.NonRetriableErrors = errs
	return &newBackoff
}

// isNonRetriableError returns true if the Error is one of NonRetriableErrors.
func (b *Backoff) isNonRetriableError(rerr *Error) bool {
	if rerr == nil {
		return false
	}

	for _, err := range b.NonRetriableErrors {
		if strings.Contains(rerr.RawError.Error(), err) {
			return true
		}
	}

	return false
}

// Step (1) returns an amount of time to sleep determined by the
// original Duration and Jitter and (2) mutates the provided Backoff
// to update its Steps and Duration.
func (b *Backoff) Step() time.Duration {
	if b.Steps < 1 {
		if b.Jitter > 0 {
			return jitter(b.Duration, b.Jitter)
		}
		return b.Duration
	}
	b.Steps--

	duration := b.Duration

	// calculate the next step
	if b.Factor != 0 {
		b.Duration = time.Duration(float64(b.Duration) * b.Factor)
		if b.Cap > 0 && b.Duration > b.Cap {
			b.Duration = b.Cap
			b.Steps = 0
		}
	}

	if b.Jitter > 0 {
		duration = jitter(duration, b.Jitter)
	}
	return duration
}

// Jitter returns a time.Duration between duration and duration + maxFactor *
// duration.
//
// This allows clients to avoid converging on periodic behavior. If maxFactor
// is 0.0, a suggested default value will be chosen.
func jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor <= 0.0 {
		maxFactor = 1.0
	}
	wait := duration + time.Duration(rand.Float64()*maxFactor*float64(duration))
	return wait
}

// DoExponentialBackoffRetry reprents an autorest.SendDecorator with backoff retry.
func DoExponentialBackoffRetry(backoff *Backoff) autorest.SendDecorator {
	return func(s autorest.Sender) autorest.Sender {
		return autorest.SenderFunc(func(r *http.Request) (*http.Response, error) {
			return doBackoffRetry(s, r, backoff)
		})
	}
}

// doBackoffRetry does the backoff retries for the request.
func doBackoffRetry(s autorest.Sender, r *http.Request, backoff *Backoff) (resp *http.Response, err error) {
	rr := autorest.NewRetriableRequest(r)
	// Increment to add the first call (attempts denotes number of retries)
	for backoff.Steps > 0 {
		err = rr.Prepare()
		if err != nil {
			return
		}
		resp, err = s.Do(rr.Request())
		rerr := GetError(resp, err)
		// Abort retries in the following scenarios:
		// 1) request succeed
		// 2) request is not retriable
		// 3) request has been throttled
		// 4) request contains non-retriable errors
		// 5) request has completed all the retry steps
		if rerr == nil || !rerr.Retriable || rerr.IsThrottled() || backoff.isNonRetriableError(rerr) || backoff.Steps == 1 {
			return resp, rerr.Error()
		}

		if !delayForBackOff(backoff, r.Context().Done()) {
			if r.Context().Err() != nil {
				return resp, r.Context().Err()
			}
			return resp, rerr.Error()
		}

		klog.V(3).Infof("Backoff retrying %s %q with error %v", r.Method, r.URL.String(), rerr)
	}

	return resp, err
}

// delayForBackOff invokes time.After for the supplied backoff duration.
// The delay may be canceled by closing the passed channel. If terminated early, returns false.
func delayForBackOff(backoff *Backoff, cancel <-chan struct{}) bool {
	d := backoff.Step()
	select {
	case <-time.After(d):
		return true
	case <-cancel:
		return false
	}
}
