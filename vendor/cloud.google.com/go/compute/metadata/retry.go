// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metadata

import (
	"context"
	"io"
	"math/rand"
	"net/http"
	"time"
)

const (
	maxRetryAttempts = 5
)

var (
	syscallRetryable = func(err error) bool { return false }
)

// defaultBackoff is basically equivalent to gax.Backoff without the need for
// the dependency.
type defaultBackoff struct {
	max time.Duration
	mul float64
	cur time.Duration
}

func (b *defaultBackoff) Pause() time.Duration {
	d := time.Duration(1 + rand.Int63n(int64(b.cur)))
	b.cur = time.Duration(float64(b.cur) * b.mul)
	if b.cur > b.max {
		b.cur = b.max
	}
	return d
}

// sleep is the equivalent of gax.Sleep without the need for the dependency.
func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	select {
	case <-ctx.Done():
		t.Stop()
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func newRetryer() *metadataRetryer {
	return &metadataRetryer{bo: &defaultBackoff{
		cur: 100 * time.Millisecond,
		max: 30 * time.Second,
		mul: 2,
	}}
}

type backoff interface {
	Pause() time.Duration
}

type metadataRetryer struct {
	bo       backoff
	attempts int
}

func (r *metadataRetryer) Retry(status int, err error) (time.Duration, bool) {
	if status == http.StatusOK {
		return 0, false
	}
	retryOk := shouldRetry(status, err)
	if !retryOk {
		return 0, false
	}
	if r.attempts == maxRetryAttempts {
		return 0, false
	}
	r.attempts++
	return r.bo.Pause(), true
}

func shouldRetry(status int, err error) bool {
	if 500 <= status && status <= 599 {
		return true
	}
	if err == io.ErrUnexpectedEOF {
		return true
	}
	// Transient network errors should be retried.
	if syscallRetryable(err) {
		return true
	}
	if err, ok := err.(interface{ Temporary() bool }); ok {
		if err.Temporary() {
			return true
		}
	}
	if err, ok := err.(interface{ Unwrap() error }); ok {
		return shouldRetry(status, err.Unwrap())
	}
	return false
}
