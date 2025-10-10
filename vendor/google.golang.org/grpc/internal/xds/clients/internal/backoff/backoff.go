/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package backoff implements the backoff strategy for clients.
//
// This is kept in internal until the clients project decides whether or not to
// allow alternative backoff strategies.
package backoff

import (
	"context"
	"errors"
	rand "math/rand/v2"
	"time"
)

// config defines the configuration options for backoff.
type config struct {
	// baseDelay is the amount of time to backoff after the first failure.
	baseDelay time.Duration
	// multiplier is the factor with which to multiply backoffs after a
	// failed retry. Should ideally be greater than 1.
	multiplier float64
	// jitter is the factor with which backoffs are randomized.
	jitter float64
	// maxDelay is the upper bound of backoff delay.
	maxDelay time.Duration
}

// defaultConfig is a backoff configuration with the default values specified
// at https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md.
//
// This should be useful for callers who want to configure backoff with
// non-default values only for a subset of the options.
var defaultConfig = config{
	baseDelay:  1.0 * time.Second,
	multiplier: 1.6,
	jitter:     0.2,
	maxDelay:   120 * time.Second,
}

// DefaultExponential is an exponential backoff implementation using the
// default values for all the configurable knobs defined in
// https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md.
var DefaultExponential = exponential{config: defaultConfig}

// exponential implements exponential backoff algorithm as defined in
// https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md.
type exponential struct {
	// Config contains all options to configure the backoff algorithm.
	config config
}

// Backoff returns the amount of time to wait before the next retry given the
// number of retries.
func (bc exponential) Backoff(retries int) time.Duration {
	if retries == 0 {
		return bc.config.baseDelay
	}
	backoff, max := float64(bc.config.baseDelay), float64(bc.config.maxDelay)
	for backoff < max && retries > 0 {
		backoff *= bc.config.multiplier
		retries--
	}
	if backoff > max {
		backoff = max
	}
	// Randomize backoff delays so that if a cluster of requests start at
	// the same time, they won't operate in lockstep.
	backoff *= 1 + bc.config.jitter*(rand.Float64()*2-1)
	if backoff < 0 {
		return 0
	}
	return time.Duration(backoff)
}

// ErrResetBackoff is the error to be returned by the function executed by RunF,
// to instruct the latter to reset its backoff state.
var ErrResetBackoff = errors.New("reset backoff state")

// RunF provides a convenient way to run a function f repeatedly until the
// context expires or f returns a non-nil error that is not ErrResetBackoff.
// When f returns ErrResetBackoff, RunF continues to run f, but resets its
// backoff state before doing so. backoff accepts an integer representing the
// number of retries, and returns the amount of time to backoff.
func RunF(ctx context.Context, f func() error, backoff func(int) time.Duration) {
	attempt := 0
	timer := time.NewTimer(0)
	for ctx.Err() == nil {
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return
		}

		err := f()
		if errors.Is(err, ErrResetBackoff) {
			timer.Reset(0)
			attempt = 0
			continue
		}
		if err != nil {
			return
		}
		timer.Reset(backoff(attempt))
		attempt++
	}
}
