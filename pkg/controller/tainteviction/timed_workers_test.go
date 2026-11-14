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

func TestNoConcurrentWorkersForSameKey(t *testing.T) {
	for name, delayed := range map[string]bool{"immediate": false, "delayed": true} {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			var calls atomic.Int32
			started := make(chan struct{}, 10)
			release := make(chan struct{})
			queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
				calls.Add(1)
				started <- struct{}{}
				<-release
				return nil
			})
			now := time.Now()
			fakeClock := testingclock.NewFakeClock(now)
			queue.clock = fakeClock
			args := NewWorkArgs("1", "1")
			key := args.KeyFromWorkArgs()

			fireAt := now
			if delayed {
				fireAt = now.Add(10 * time.Second)
			}
			queue.AddWork(ctx, args, now, fireAt)
			fakeClock.Step(11 * time.Second)
			<-started

			if queue.CancelWork(logger, key) {
				t.Error("CancelWork reported success for a running worker")
			}
			queue.AddWork(ctx, args, now, now)
			queue.UpdateWork(ctx, args, now, now.Add(time.Hour))
			close(release)
			queue.CancelAndWait()

			if got := calls.Load(); got != 1 {
				t.Errorf("expected 1 call while the first worker was running, got %d", got)
			}
		})
	}
}

func TestWorkerRemovesItselfWhenDone(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	done := make(chan struct{}, 2)
	queue := CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		done <- struct{}{}
		return nil
	})
	args := NewWorkArgs("1", "1")
	now := time.Now()
	queue.AddWork(ctx, args, now, now)
	<-done
	// The entry is removed after the work function returns, so poll for it.
	for {
		queue.Lock()
		_, exists := queue.workers[args.KeyFromWorkArgs()]
		queue.Unlock()
		if !exists {
			break
		}
		time.Sleep(time.Millisecond)
	}
	queue.AddWork(ctx, args, now, now)
	<-done
	queue.CancelAndWait()
}
