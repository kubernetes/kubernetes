// Copyright 2016, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package gax

import (
	"context"
	"errors"
	"testing"
	"time"
)

var canceledContext context.Context

func init() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	canceledContext = ctx
}

// recordSleeper is a test implementation of sleeper.
type recordSleeper int

func (s *recordSleeper) sleep(ctx context.Context, _ time.Duration) error {
	*s++
	return ctx.Err()
}

type boolRetryer bool

func (r boolRetryer) Retry(err error) (time.Duration, bool) { return 0, bool(r) }

func TestInvokeSuccess(t *testing.T) {
	apiCall := func(context.Context, CallSettings) error { return nil }
	var sp recordSleeper
	err := invoke(context.Background(), apiCall, CallSettings{}, sp.sleep)

	if err != nil {
		t.Errorf("found error %s, want nil", err)
	}
	if sp != 0 {
		t.Errorf("slept %d times, should not have slept since the call succeeded", int(sp))
	}
}

func TestInvokeNoRetry(t *testing.T) {
	apiErr := errors.New("foo error")
	apiCall := func(context.Context, CallSettings) error { return apiErr }
	var sp recordSleeper
	err := invoke(context.Background(), apiCall, CallSettings{}, sp.sleep)

	if err != apiErr {
		t.Errorf("found error %s, want %s", err, apiErr)
	}
	if sp != 0 {
		t.Errorf("slept %d times, should not have slept since retry is not specified", int(sp))
	}
}

func TestInvokeNilRetry(t *testing.T) {
	apiErr := errors.New("foo error")
	apiCall := func(context.Context, CallSettings) error { return apiErr }
	var settings CallSettings
	WithRetry(func() Retryer { return nil }).Resolve(&settings)
	var sp recordSleeper
	err := invoke(context.Background(), apiCall, settings, sp.sleep)

	if err != apiErr {
		t.Errorf("found error %s, want %s", err, apiErr)
	}
	if sp != 0 {
		t.Errorf("slept %d times, should not have slept since retry is not specified", int(sp))
	}
}

func TestInvokeNeverRetry(t *testing.T) {
	apiErr := errors.New("foo error")
	apiCall := func(context.Context, CallSettings) error { return apiErr }
	var settings CallSettings
	WithRetry(func() Retryer { return boolRetryer(false) }).Resolve(&settings)
	var sp recordSleeper
	err := invoke(context.Background(), apiCall, settings, sp.sleep)

	if err != apiErr {
		t.Errorf("found error %s, want %s", err, apiErr)
	}
	if sp != 0 {
		t.Errorf("slept %d times, should not have slept since retry is not specified", int(sp))
	}
}

func TestInvokeRetry(t *testing.T) {
	const target = 3

	retryNum := 0
	apiErr := errors.New("foo error")
	apiCall := func(context.Context, CallSettings) error {
		retryNum++
		if retryNum < target {
			return apiErr
		}
		return nil
	}
	var settings CallSettings
	WithRetry(func() Retryer { return boolRetryer(true) }).Resolve(&settings)
	var sp recordSleeper
	err := invoke(context.Background(), apiCall, settings, sp.sleep)

	if err != nil {
		t.Errorf("found error %s, want nil, call should have succeeded after %d tries", err, target)
	}
	if sp != target-1 {
		t.Errorf("retried %d times, want %d", int(sp), int(target-1))
	}
}

func TestInvokeRetryTimeout(t *testing.T) {
	apiErr := errors.New("foo error")
	apiCall := func(context.Context, CallSettings) error { return apiErr }
	var settings CallSettings
	WithRetry(func() Retryer { return boolRetryer(true) }).Resolve(&settings)
	var sp recordSleeper

	err := invoke(canceledContext, apiCall, settings, sp.sleep)

	if err != context.Canceled {
		t.Errorf("found error %s, want %s", err, context.Canceled)
	}
}
