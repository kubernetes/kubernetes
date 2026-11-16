/*
Copyright The Kubernetes Authors.

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

package cacher

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/utils/clock"
)

func TestCurrentResourceVersionBatcher(t *testing.T) {
	type outcome struct {
		rv       uint64
		delay time.Duration
		err      error
	}
	type request struct {
		at          time.Duration
		timeout     time.Duration
		wantOutcome outcome
	}

	errStorage := errors.New("storage failure")

	tests := []struct {
		name         string
		readDuration time.Duration
		readErr      error
		requests     []request
		wantReads    uint64
	}{
		{
			name: "single request returns immediately",
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 0}},
			},
			wantReads: 1,
		},
		{
			name: "requests spaced by at least 5ms return immediately",
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 0}},
				{at: 5 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 0}},
				{at: 12 * time.Millisecond, wantOutcome: outcome{rv: 3, delay: 0}},
			},
			wantReads: 3,
		},
		{
			name: "requests within 5ms wait until 5ms and batch together",
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 0}},
				{at: time.Millisecond, wantOutcome: outcome{rv: 2, delay: 4 * time.Millisecond}},
				{at: time.Millisecond, wantOutcome: outcome{rv: 2, delay: 4 * time.Millisecond}},
				{at: 2 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 3 * time.Millisecond}},
				{at: 4 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: time.Millisecond}},
				{at: 6 * time.Millisecond, wantOutcome: outcome{rv: 3, delay: 4 * time.Millisecond}},
			},
			wantReads: 3,
		},
		{
			name:         "callers arriving during in-flight read join next batch",
			readDuration: 8 * time.Millisecond,
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 8 * time.Millisecond}},
				{at: 2 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 11 * time.Millisecond}},
				{at: 4 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 9 * time.Millisecond}},
			},
			wantReads: 2,
		},
		{
			name:         "read shorter than 5ms waits remaining interval from read start",
			readDuration: 2 * time.Millisecond,
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 2 * time.Millisecond}},
				{at: time.Millisecond, wantOutcome: outcome{rv: 2, delay: 6 * time.Millisecond}},
				{at: 3 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 4 * time.Millisecond}},
			},
			wantReads: 2,
		},
		{
			name:    "storage error propagates to batched callers",
			readErr: errStorage,
			requests: []request{
				{at: 0, wantOutcome: outcome{err: errStorage, delay: 0}},
				{at: time.Millisecond, wantOutcome: outcome{err: errStorage, delay: 4 * time.Millisecond}},
				{at: 2 * time.Millisecond, wantOutcome: outcome{err: errStorage, delay: 3 * time.Millisecond}},
				{at: 10 * time.Millisecond, wantOutcome: outcome{err: errStorage, delay: 0}},
			},
			wantReads: 3,
		},
		{
			name:         "caller timeout does not affect remaining batched callers",
			readDuration: 4 * time.Millisecond,
			requests: []request{
				{at: 0, wantOutcome: outcome{rv: 1, delay: 4 * time.Millisecond}},
				{at: time.Millisecond, timeout: 2 * time.Millisecond, wantOutcome: outcome{err: context.DeadlineExceeded, delay: 2 * time.Millisecond}},
				{at: 2 * time.Millisecond, timeout: 5 * time.Millisecond, wantOutcome: outcome{err: context.DeadlineExceeded, delay: 5 * time.Millisecond}},
				{at: 3 * time.Millisecond, wantOutcome: outcome{rv: 2, delay: 6 * time.Millisecond}},
			},
			wantReads: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var reads atomic.Uint64
				batcher := newCurrentResourVersionBatcher(func(_ context.Context) (uint64, error) {
					rv := reads.Add(1)
					if tc.readDuration > 0 {
						time.Sleep(tc.readDuration)
					}
					if tc.readErr != nil {
						return 0, tc.readErr
					}
					return rv, nil
				}, clock.RealClock{})

				start := time.Now()
				outcomes := make([]outcome, len(tc.requests))
				var wg sync.WaitGroup

				t.Log("Execute requests")
				for i, req := range tc.requests {
					elapsed := time.Since(start)
					require.GreaterOrEqual(t, req.at, elapsed)
					if req.at > elapsed {
						time.Sleep(req.at - elapsed)
					}
					wg.Add(1)
					go func() {
						defer wg.Done()
						ctx := context.Background()
						if req.timeout > 0 {
							var cancel context.CancelFunc
							ctx, cancel = context.WithTimeout(ctx, req.timeout)
							defer cancel()
						}
						callStart := time.Now()
						rv, err := batcher.GetCurrentResourceVersion(ctx)
						outcomes[i] = outcome{
							rv:       rv,
							delay: time.Since(callStart),
							err:      err,
						}
					}()
					synctest.Wait()
				}

				t.Log("Wait for requests and verify outcomes")
				wg.Wait()

				assert.Equal(t, tc.wantReads, reads.Load())
				for i, req := range tc.requests {
					assert.Equal(t, req.wantOutcome, outcomes[i], "request %d (at=%v)", i, req.at)
				}
			})
		})
	}
}
