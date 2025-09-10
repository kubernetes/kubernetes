// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"net/http"
	"time"

	"github.com/google/go-containerregistry/internal/retry"
)

// Sleep for 0.1 then 0.3 seconds. This should cover networking blips.
var defaultBackoff = retry.Backoff{
	Duration: 100 * time.Millisecond,
	Factor:   3.0,
	Jitter:   0.1,
	Steps:    3,
}

var _ http.RoundTripper = (*retryTransport)(nil)

// retryTransport wraps a RoundTripper and retries temporary network errors.
type retryTransport struct {
	inner     http.RoundTripper
	backoff   retry.Backoff
	predicate retry.Predicate
	codes     []int
}

// Option is a functional option for retryTransport.
type Option func(*options)

type options struct {
	backoff   retry.Backoff
	predicate retry.Predicate
	codes     []int
}

// Backoff is an alias of retry.Backoff to expose this configuration option to consumers of this lib
type Backoff = retry.Backoff

// WithRetryBackoff sets the backoff for retry operations.
func WithRetryBackoff(backoff Backoff) Option {
	return func(o *options) {
		o.backoff = backoff
	}
}

// WithRetryPredicate sets the predicate for retry operations.
func WithRetryPredicate(predicate func(error) bool) Option {
	return func(o *options) {
		o.predicate = predicate
	}
}

// WithRetryStatusCodes sets which http response codes will be retried.
func WithRetryStatusCodes(codes ...int) Option {
	return func(o *options) {
		o.codes = codes
	}
}

// NewRetry returns a transport that retries errors.
func NewRetry(inner http.RoundTripper, opts ...Option) http.RoundTripper {
	o := &options{
		backoff:   defaultBackoff,
		predicate: retry.IsTemporary,
	}

	for _, opt := range opts {
		opt(o)
	}

	return &retryTransport{
		inner:     inner,
		backoff:   o.backoff,
		predicate: o.predicate,
		codes:     o.codes,
	}
}

func (t *retryTransport) RoundTrip(in *http.Request) (out *http.Response, err error) {
	roundtrip := func() error {
		out, err = t.inner.RoundTrip(in)
		if !retry.Ever(in.Context()) {
			return nil
		}
		if out != nil {
			for _, code := range t.codes {
				if out.StatusCode == code {
					return retryError(out)
				}
			}
		}
		return err
	}
	retry.Retry(roundtrip, t.predicate, t.backoff)
	return
}
