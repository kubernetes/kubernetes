// Copyright 2021 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"errors"
	"io"
	"net"
	"strings"
	"time"

	"github.com/googleapis/gax-go/v2"
	"google.golang.org/api/googleapi"
)

// Backoff is an interface around gax.Backoff's Pause method, allowing tests to provide their
// own implementation.
type Backoff interface {
	Pause() time.Duration
}

// These are declared as global variables so that tests can overwrite them.
var (
	// Default per-chunk deadline for resumable uploads.
	defaultRetryDeadline = 32 * time.Second
	// Default backoff timer.
	backoff = func() Backoff {
		return &gax.Backoff{Initial: 100 * time.Millisecond}
	}
	// syscallRetryable is a platform-specific hook, specified in retryable_linux.go
	syscallRetryable func(error) bool = func(err error) bool { return false }
)

const (
	// statusTooManyRequests is returned by the storage API if the
	// per-project limits have been temporarily exceeded. The request
	// should be retried.
	// https://cloud.google.com/storage/docs/json_api/v1/status-codes#standardcodes
	statusTooManyRequests = 429

	// statusRequestTimeout is returned by the storage API if the
	// upload connection was broken. The request should be retried.
	statusRequestTimeout = 408
)

// shouldRetry indicates whether an error is retryable for the purposes of this
// package, unless a ShouldRetry func is specified by the RetryConfig instead.
// It follows guidance from
// https://cloud.google.com/storage/docs/exponential-backoff .
func shouldRetry(status int, err error) bool {
	if 500 <= status && status <= 599 {
		return true
	}
	if status == statusTooManyRequests || status == statusRequestTimeout {
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
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if strings.Contains(opErr.Error(), "use of closed network connection") {
			// TODO: check against net.ErrClosed (go 1.16+) instead of string
			return true
		}
	}

	// If Go 1.13 error unwrapping is available, use this to examine wrapped
	// errors.
	if err, ok := err.(interface{ Unwrap() error }); ok {
		return shouldRetry(status, err.Unwrap())
	}
	return false
}

// RetryConfig allows configuration of backoff timing and retryable errors.
type RetryConfig struct {
	Backoff     *gax.Backoff
	ShouldRetry func(err error) bool
}

// Get a new backoff object based on the configured values.
func (r *RetryConfig) backoff() Backoff {
	if r == nil || r.Backoff == nil {
		return backoff()
	}
	return &gax.Backoff{
		Initial:    r.Backoff.Initial,
		Max:        r.Backoff.Max,
		Multiplier: r.Backoff.Multiplier,
	}
}

// This is kind of hacky; it is necessary because ShouldRetry expects to
// handle HTTP errors via googleapi.Error, but the error has not yet been
// wrapped with a googleapi.Error at this layer, and the ErrorFunc type
// in the manual layer does not pass in a status explicitly as it does
// here. So, we must wrap error status codes in a googleapi.Error so that
// ShouldRetry can parse this correctly.
func (r *RetryConfig) errorFunc() func(status int, err error) bool {
	if r == nil || r.ShouldRetry == nil {
		return shouldRetry
	}
	return func(status int, err error) bool {
		if status >= 400 {
			return r.ShouldRetry(&googleapi.Error{Code: status})
		}
		return r.ShouldRetry(err)
	}
}
