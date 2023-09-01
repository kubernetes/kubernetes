/*
 *
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package retry provides a retry helper for talking to S2A gRPC server.
// The implementation is modeled after
// https://github.com/googleapis/google-cloud-go/blob/main/compute/metadata/retry.go
package retry

import (
	"context"
	"math/rand"
	"time"

	"google.golang.org/grpc/grpclog"
)

const (
	maxRetryAttempts = 5
	maxRetryForLoops = 10
)

type defaultBackoff struct {
	max time.Duration
	mul float64
	cur time.Duration
}

// Pause returns a duration, which is used as the backoff wait time
// before the next retry.
func (b *defaultBackoff) Pause() time.Duration {
	d := time.Duration(1 + rand.Int63n(int64(b.cur)))
	b.cur = time.Duration(float64(b.cur) * b.mul)
	if b.cur > b.max {
		b.cur = b.max
	}
	return d
}

// Sleep will wait for the specified duration or return on context
// expiration.
func Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	select {
	case <-ctx.Done():
		t.Stop()
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// NewRetryer creates an instance of S2ARetryer using the defaultBackoff
// implementation.
var NewRetryer = func() *S2ARetryer {
	return &S2ARetryer{bo: &defaultBackoff{
		cur: 100 * time.Millisecond,
		max: 30 * time.Second,
		mul: 2,
	}}
}

type backoff interface {
	Pause() time.Duration
}

// S2ARetryer implements a retry helper for talking to S2A gRPC server.
type S2ARetryer struct {
	bo       backoff
	attempts int
}

// Attempts return the number of retries attempted.
func (r *S2ARetryer) Attempts() int {
	return r.attempts
}

// Retry returns a boolean indicating whether retry should be performed
// and the backoff duration.
func (r *S2ARetryer) Retry(err error) (time.Duration, bool) {
	if err == nil {
		return 0, false
	}
	if r.attempts >= maxRetryAttempts {
		return 0, false
	}
	r.attempts++
	return r.bo.Pause(), true
}

// Run uses S2ARetryer to execute the function passed in, until success or reaching
// max number of retry attempts.
func Run(ctx context.Context, f func() error) {
	retryer := NewRetryer()
	forLoopCnt := 0
	var err error
	for {
		err = f()
		if bo, shouldRetry := retryer.Retry(err); shouldRetry {
			if grpclog.V(1) {
				grpclog.Infof("will attempt retry: %v", err)
			}
			if ctx.Err() != nil {
				if grpclog.V(1) {
					grpclog.Infof("exit retry loop due to context error: %v", ctx.Err())
				}
				break
			}
			if sleepErr := Sleep(ctx, bo); sleepErr != nil {
				if grpclog.V(1) {
					grpclog.Infof("exit retry loop due to sleep error: %v", sleepErr)
				}
				break
			}
			// This shouldn't happen, just make sure we are not stuck in the for loops.
			forLoopCnt++
			if forLoopCnt > maxRetryForLoops {
				if grpclog.V(1) {
					grpclog.Infof("exit the for loop after too many retries")
				}
				break
			}
			continue
		}
		if grpclog.V(1) {
			grpclog.Infof("retry conditions not met, exit the loop")
		}
		break
	}
}
