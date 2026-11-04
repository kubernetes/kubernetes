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
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/progress"
	"k8s.io/utils/clock"
)

func TestWaitUntilFreshLocked(t *testing.T) {
	type outcome struct {
		rv       uint64
		duration time.Duration
		err      error
	}
	type waiter struct {
		requestRV   uint64
		wantOutcome outcome
	}
	type step struct {
		at       time.Duration
		updateRV uint64
		waiter   *waiter
	}

	tooLargeErr := func(requestRV, currentRV uint64) error {
		return storage.NewTooLargeResourceVersionError(requestRV, currentRV, resourceVersionTooHighRetrySeconds)
	}

	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "rv=0 on uninitialized cache returns immediately",
			steps: []step{
				{waiter: &waiter{requestRV: 0, wantOutcome: outcome{rv: 0, duration: 0}}},
			},
		},
		{
			name: "past and current rv on initialized cache return immediately",
			steps: []step{
				{updateRV: 10},
				{waiter: &waiter{requestRV: 0, wantOutcome: outcome{rv: 10, duration: 0}}},
				{waiter: &waiter{requestRV: 5, wantOutcome: outcome{rv: 10, duration: 0}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 10, duration: 0}}},
			},
		},
		{
			name: "single waiter wakes on exact match",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 10, duration: 100 * time.Millisecond}}},
				{at: 100 * time.Millisecond, updateRV: 10},
			},
		},
		{
			name: "single waiter wakes on higher match",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 15, duration: 100 * time.Millisecond}}},
				{at: 100 * time.Millisecond, updateRV: 15},
			},
		},
		{
			name: "lower and duplicate updates keep waiter blocked until timeout",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 9, duration: blockTimeout, err: tooLargeErr(10, 9)}}},
				{at: 100 * time.Millisecond, updateRV: 7},
				{at: 200 * time.Millisecond, updateRV: 9},
				{at: 300 * time.Millisecond, updateRV: 9},
			},
		},
		{
			name: "lower and duplicate updates followed by target rv wake waiter",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 10, duration: 400 * time.Millisecond}}},
				{at: 100 * time.Millisecond, updateRV: 7},
				{at: 200 * time.Millisecond, updateRV: 9},
				{at: 300 * time.Millisecond, updateRV: 9},
				{at: 400 * time.Millisecond, updateRV: 10},
			},
		},
		{
			name: "timeout with no updates",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 5, duration: blockTimeout, err: tooLargeErr(10, 5)}}},
			},
		},
		{
			name: "non-ordered waiters wake selectively at matching rv and time",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 25, wantOutcome: outcome{rv: 15, duration: blockTimeout, err: tooLargeErr(25, 15)}}},
				{waiter: &waiter{requestRV: 0, wantOutcome: outcome{rv: 5, duration: 0}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 12, duration: 200 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 5, wantOutcome: outcome{rv: 5, duration: 0}}},
				{waiter: &waiter{requestRV: 15, wantOutcome: outcome{rv: 15, duration: 300 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 6, wantOutcome: outcome{rv: 7, duration: 100 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 12, duration: 200 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 20, wantOutcome: outcome{rv: 15, duration: blockTimeout, err: tooLargeErr(20, 15)}}},
				{at: 100 * time.Millisecond, updateRV: 7},
				{at: 200 * time.Millisecond, updateRV: 12},
				{at: 300 * time.Millisecond, updateRV: 15},
			},
		},
		{
			name: "non-ordered waiters all wake when rv jumps past max",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 25, wantOutcome: outcome{rv: 30, duration: 100 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 30, duration: 100 * time.Millisecond}}},
				{waiter: &waiter{requestRV: 15, wantOutcome: outcome{rv: 30, duration: 100 * time.Millisecond}}},
				{at: 100 * time.Millisecond, updateRV: 30},
			},
		},
		{
			name: "non-ordered waiters all time out",
			steps: []step{
				{updateRV: 5},
				{waiter: &waiter{requestRV: 20, wantOutcome: outcome{rv: 8, duration: blockTimeout, err: tooLargeErr(20, 8)}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 8, duration: blockTimeout, err: tooLargeErr(10, 8)}}},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 8, duration: blockTimeout, err: tooLargeErr(10, 8)}}},
				{at: 100 * time.Millisecond, updateRV: 8},
			},
		},
		{
			name: "staggered waiters have independent deadlines",
			steps: []step{
				{updateRV: 2},
				{waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 6, duration: blockTimeout, err: tooLargeErr(10, 6)}}},
				{at: time.Second, updateRV: 6},
				{at: time.Second, waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 10, duration: 2500 * time.Millisecond}}},
				{at: 3500 * time.Millisecond, updateRV: 10},
			},
		},
		{
			name: "waiter registered after earlier waiter on same rv timed out still wakes",
			steps: []step{
				{updateRV: 2},
				{waiter: &waiter{requestRV: 15, wantOutcome: outcome{rv: 2, duration: blockTimeout, err: tooLargeErr(15, 2)}}},
				{at: 500 * time.Millisecond, waiter: &waiter{requestRV: 10, wantOutcome: outcome{rv: 15, duration: 2700 * time.Millisecond}}},
				{at: 600 * time.Millisecond, waiter: &waiter{requestRV: 20, wantOutcome: outcome{rv: 15, duration: blockTimeout, err: tooLargeErr(20, 15)}}},
				{at: blockTimeout + 100*time.Millisecond, waiter: &waiter{requestRV: 15, wantOutcome: outcome{rv: 15, duration: 100 * time.Millisecond}}},
				{at: blockTimeout + 200*time.Millisecond, updateRV: 15},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Cleanup(metrics.WatchCacheReadWait.Reset)
			synctest.Test(t, func(t *testing.T) {
				fw := newTestFreshnessWaiter()
				go fw.w.Run(t.Context().Done())
				start := time.Now()
				outcomes := make([]outcome, len(tc.steps))
				wg := sync.WaitGroup{}

				t.Log("Execute steps")
				for i, s := range tc.steps {
					elapsed := time.Since(start)
					require.GreaterOrEqual(t, s.at, elapsed)
					if s.at > elapsed {
						time.Sleep(s.at - elapsed)
					}
					switch {
					case s.updateRV > 0 && s.waiter == nil:
						fw.updateRV(s.updateRV)
					case s.waiter != nil && s.updateRV == 0:
						wg.Add(1)
						go func() {
							defer wg.Done()
							waitStart := time.Now()
							rv, err := fw.waitUntilFresh(context.Background(), s.waiter.requestRV)
							outcomes[i] = outcome{
								rv:       rv,
								duration: time.Since(waitStart),
								err:      err,
							}
						}()
					default:
						t.Fatalf("step %d must set either updateRV or waiter", i)
					}
					synctest.Wait()
				}

				t.Log("Advance time by blockTimeout and verify outcomes")
				time.Sleep(blockTimeout)
				wg.Wait()

				for i, s := range tc.steps {
					if s.waiter != nil {
						assert.Equal(t, s.waiter.wantOutcome, outcomes[i], "step %d (rv=%d)", i, s.waiter.requestRV)
					}
				}
			})
		})
	}
}

type testFreshnessWaiter struct {
	mu sync.RWMutex
	w  *freshnessWaiter
}

func newTestFreshnessWaiter() *testFreshnessWaiter {
	c := clock.RealClock{}
	pr := progress.NewConditionalProgressRequester(func(context.Context) error { return nil }, c, nil)
	tw := &testFreshnessWaiter{}
	tw.w = newFreshnessWaiter(c, schema.GroupResource{Resource: "pods"}, pr)
	return tw
}

func (tw *testFreshnessWaiter) waitUntilFresh(ctx context.Context, resourceVersion uint64) (uint64, error) {
	tw.mu.RLock()
	defer tw.mu.RUnlock()
	err := tw.w.waitUntilFreshLockedWithLocker(ctx, resourceVersion, tw.mu.RLocker())
	return tw.w.latestRV, err
}

func (tw *testFreshnessWaiter) updateRV(rv uint64) {
	tw.mu.Lock()
	defer tw.mu.Unlock()
	tw.w.broadcast(rv)
}

func BenchmarkWaitUntilFreshLocked(b *testing.B) {
	b.Cleanup(metrics.WatchCacheReadWait.Reset)
	for _, waiterCount := range []int{500} {
		b.Run(fmt.Sprintf("Waiters=%d", waiterCount), func(b *testing.B) {
			maxRV := uint64(waiterCount)
			for _, waiterDistribution := range []string{"late", "uniform", "mixed"} {
				waiterResourceVersions := waiterResourceVersions(waiterCount, maxRV, waiterDistribution)
				b.Run(fmt.Sprintf("WaitersDistribution=%s", waiterDistribution), func(b *testing.B) {
					for _, updateDistribution := range []string{"dense", "coarse", "single", "mixed"} {
						resourceVersionUpdates := resourceVersionUpdates(maxRV, updateDistribution)
						b.Run(fmt.Sprintf("ResourceVersionDistribution=%s", updateDistribution), func(b *testing.B) {
							runBenchmarkWaitUntilFresh(b, waiterResourceVersions, resourceVersionUpdates, maxRV)
						})
					}
				})
			}
		})
	}
}

func runBenchmarkWaitUntilFresh(b *testing.B, waiterResourceVersions, resourceVersionUpdates []uint64, maxRV uint64) {
	b.ReportAllocs()
	ctx := b.Context()
	fw := newTestFreshnessWaiter()
	go fw.w.Run(ctx.Done())

	var stopWaiters atomic.Bool
	var remainingWaiters atomic.Int64
	remainingWaiters.Store(int64(len(waiterResourceVersions)))
	waiterWokenUp := make(chan struct{}, 1)
	var wg sync.WaitGroup
	for _, rvOffset := range waiterResourceVersions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for baseRV := uint64(0); ; baseRV += maxRV {
				if stopWaiters.Load() {
					return
				}
				if remainingWaiters.Add(-1) == 0 {
					waiterWokenUp <- struct{}{}
				}
				if _, err := fw.waitUntilFresh(ctx, baseRV+rvOffset); err != nil {
					b.Errorf("unexpected error: %v", err)
					return
				}
			}
		}()
	}
	<-waiterWokenUp

	var baseRV uint64
	for b.Loop() {
		remainingWaiters.Store(int64(len(waiterResourceVersions)))
		for _, step := range resourceVersionUpdates {
			fw.updateRV(baseRV + step)
		}
		<-waiterWokenUp
		baseRV += maxRV
	}

	stopWaiters.Store(true)
	fw.updateRV(math.MaxUint64)
	wg.Wait()
}

func waiterResourceVersions(count int, maxRV uint64, dist string) []uint64 {
	offsets := make([]uint64, count)
	for i := range count {
		switch dist {
		case "late":
			offsets[i] = maxRV
		case "uniform":
			offsets[i] = uint64(i*37)%maxRV + 1
		case "mixed":
			if i < count/2 {
				offsets[i] = uint64(i*37)%maxRV + 1
			} else {
				offsets[i] = maxRV
			}
		default:
			panic(fmt.Sprintf("unknown target distribution: %s", dist))
		}
	}
	return offsets
}

func resourceVersionUpdates(maxRV uint64, pattern string) []uint64 {
	switch pattern {
	case "dense":
		steps := make([]uint64, maxRV)
		for i := range maxRV {
			steps[i] = i + 1
		}
		return steps
	case "coarse":
		stride := max(maxRV/5, 1)
		var steps []uint64
		for rv := stride; rv < maxRV; rv += stride {
			steps = append(steps, rv)
		}
		return append(steps, maxRV)
	case "single":
		return []uint64{maxRV}
	case "mixed":
		chunk := max(maxRV/5, 1)
		denseRun := max(chunk/5, 1)
		var steps []uint64
		for base := uint64(0); base < maxRV; base += chunk {
			for d := uint64(1); d <= denseRun && base+d < base+chunk; d++ {
				steps = append(steps, base+d)
			}
			steps = append(steps, min(base+chunk, maxRV))
		}
		return steps
	default:
		panic(fmt.Sprintf("unknown update pattern: %s", pattern))
	}
}
