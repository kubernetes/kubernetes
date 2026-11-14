/*
Copyright 2017 The Kubernetes Authors.

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

package tainteviction

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/klog/v2/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

func TestExecute(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(5)
	queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, now)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, now)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, now)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, now)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, now)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 5 {
		t.Errorf("Expected testVal = 5, got %v", lastVal)
	}
}

func TestExecuteDelayed(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(5)
	queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(10 * time.Second)
	fakeClock := testingclock.NewFakeClock(now)
	queue.clock = fakeClock
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	fakeClock.Step(11 * time.Second)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 5 {
		t.Errorf("Expected testVal = 5, got %v", lastVal)
	}
}

func TestCancel(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(3)
	queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(10 * time.Second)
	fakeClock := testingclock.NewFakeClock(now)
	queue.clock = fakeClock
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	queue.CancelWork(logger, NewWorkArgs("2", "2").KeyFromWorkArgs())
	queue.CancelWork(logger, NewWorkArgs("4", "4").KeyFromWorkArgs())
	fakeClock.Step(11 * time.Second)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 3 {
		t.Errorf("Expected testVal = 3, got %v", lastVal)
	}
}

func TestCancelAndRead(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(4)
	queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(10 * time.Second)
	fakeClock := testingclock.NewFakeClock(now)
	queue.clock = fakeClock
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	queue.AddWork(ctx, NewWorkArgs("1", "1"), now, then)
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	queue.AddWork(ctx, NewWorkArgs("3", "3"), now, then)
	queue.AddWork(ctx, NewWorkArgs("4", "4"), now, then)
	queue.AddWork(ctx, NewWorkArgs("5", "5"), now, then)
	queue.CancelWork(logger, NewWorkArgs("2", "2").KeyFromWorkArgs())
	queue.CancelWork(logger, NewWorkArgs("4", "4").KeyFromWorkArgs())
	queue.AddWork(ctx, NewWorkArgs("2", "2"), now, then)
	fakeClock.Step(11 * time.Second)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 4 {
		t.Errorf("Expected testVal = 4, got %v", lastVal)
	}
}

func TestRunningWorkerCompletionDoesNotRemoveReplacement(t *testing.T) {
	testCases := []struct {
		name                  string
		startFirstWorker      func(context.Context, *TimedWorkerQueue, *testingclock.FakeClock, time.Time, *WorkArgs)
		expectCancelToSucceed bool
	}{
		{
			name: "immediate worker",
			startFirstWorker: func(ctx context.Context, queue *TimedWorkerQueue, fakeClock *testingclock.FakeClock, now time.Time, args *WorkArgs) {
				queue.AddWork(ctx, args, now, now)
			},
			expectCancelToSucceed: false,
		},
		{
			name: "scheduled worker already firing",
			startFirstWorker: func(ctx context.Context, queue *TimedWorkerQueue, fakeClock *testingclock.FakeClock, now time.Time, args *WorkArgs) {
				queue.AddWork(ctx, args, now, now.Add(time.Second))
				fakeClock.Step(time.Second)
			},
			expectCancelToSucceed: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			now := time.Now()
			fakeClock := testingclock.NewFakeClock(now)
			started := make(chan struct{})
			release := make(chan struct{})
			startOnce := sync.Once{}

			queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
				startOnce.Do(func() {
					close(started)
				})
				<-release
				return nil
			})
			queue.clock = fakeClock

			args := NewWorkArgs("pod", "namespace")
			key := args.KeyFromWorkArgs()
			tc.startFirstWorker(ctx, queue, fakeClock, now, args)
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("timed worker did not start")
			}

			if got := queue.CancelWork(logger, key); got != tc.expectCancelToSucceed {
				t.Fatalf("CancelWork() = %v, want %v", got, tc.expectCancelToSucceed)
			}

			replacementFireAt := now.Add(10 * time.Second)
			queue.AddWork(ctx, NewWorkArgs("pod", "namespace"), now.Add(2*time.Second), replacementFireAt)
			replacement := queue.GetWorkerUnsafe(key)
			if replacement == nil {
				t.Fatal("replacement worker was not added")
			}
			if replacement.FireAt != replacementFireAt {
				t.Fatalf("replacement worker FireAt = %v, want %v", replacement.FireAt, replacementFireAt)
			}

			close(release)
			queue.workerWG.Wait()

			replacement = queue.GetWorkerUnsafe(key)
			if replacement == nil {
				t.Fatal("running worker removed replacement worker on completion")
			}
			if replacement.FireAt != replacementFireAt {
				t.Fatalf("replacement worker FireAt after stale completion = %v, want %v", replacement.FireAt, replacementFireAt)
			}
			if !queue.CancelWork(logger, key) {
				t.Fatal("replacement worker was not cancellable after stale worker completed")
			}
			if got := queue.GetWorkerUnsafe(key); got != nil {
				t.Fatalf("worker still present after cancellation: %#v", got)
			}
		})
	}
}
