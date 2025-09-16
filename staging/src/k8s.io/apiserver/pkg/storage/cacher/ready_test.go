/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func Test_newReady(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	ready.setError(nil)
	// create 10 goroutines waiting for ready
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait(context.Background())
		}()
	}
	select {
	case <-time.After(1 * time.Second):
	case <-errCh:
		t.Errorf("ready should be blocking")
	}
	ready.setReady()
	for i := 0; i < 10; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("unexpected error on channel %d", i)
		}
	}
}

func Test_newReadySetIdempotent(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	ready.setError(nil)
	ready.setError(nil)
	ready.setError(nil)
	if generation, _, err := ready.checkAndReadGeneration(); generation != 0 || err == nil {
		t.Errorf("unexpected state: generation=%v ready=%v", generation, err)
	}
	ready.setReady()
	if generation, _, err := ready.checkAndReadGeneration(); generation != 1 || err != nil {
		t.Errorf("unexpected state: generation=%v ready=%v", generation, err)
	}
	ready.setReady()
	ready.setReady()
	if generation, _, err := ready.checkAndReadGeneration(); generation != 1 || err != nil {
		t.Errorf("unexpected state: generation=%v ready=%v", generation, err)
	}
	ready.setError(nil)
	// create 10 goroutines waiting for ready and stop
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait(context.Background())
		}()
	}
	select {
	case <-time.After(1 * time.Second):
	case <-errCh:
		t.Errorf("ready should be blocking")
	}
	ready.setReady()
	if generation, _, err := ready.checkAndReadGeneration(); generation != 2 || err != nil {
		t.Errorf("unexpected state: generation=%v ready=%v", generation, err)
	}
	for i := 0; i < 10; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("unexpected error on channel %d", i)
		}
	}
}

// Test_newReadyRacy executes all the possible transitions randomly.
// It must run with the race detector enabled.
func Test_newReadyRacy(t *testing.T) {
	concurrency := 1000
	errCh := make(chan error, concurrency)
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	ready.setError(nil)

	wg := sync.WaitGroup{}
	wg.Add(2 * concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			errCh <- ready.wait(context.Background())
		}()
		go func() {
			defer wg.Done()
			ready.setError(nil)
		}()
		go func() {
			defer wg.Done()
			ready.setReady()
		}()
	}
	// Last one has to be set to true.
	wg.Wait()
	ready.setReady()

	for i := 0; i < concurrency; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("unexpected error %v on channel %d", err, i)
		}
	}
}

func Test_newReadyStop(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	ready.setError(nil)
	// create 10 goroutines waiting for ready and stop
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait(context.Background())
		}()
	}
	select {
	case <-time.After(1 * time.Second):
	case <-errCh:
		t.Errorf("ready should be blocking")
	}
	ready.stop()
	for i := 0; i < 10; i++ {
		if err := <-errCh; err == nil {
			t.Errorf("unexpected success on channel %d", i)
		}
	}
}

func Test_newReadyCheck(t *testing.T) {
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	// it starts as false
	if _, err := ready.check(); err == nil {
		t.Errorf("unexpected ready state %v", err)
	}
	ready.setReady()
	if _, err := ready.check(); err != nil {
		t.Errorf("unexpected ready state %v", err)
	}
	// stop sets ready to false
	ready.stop()
	if _, err := ready.check(); err == nil {
		t.Errorf("unexpected ready state %v", err)
	}
	// can not set to true if is stopped
	ready.setReady()
	if _, err := ready.check(); err == nil {
		t.Errorf("unexpected ready state %v", err)
	}
	err := ready.wait(context.Background())
	if err == nil {
		t.Errorf("expected error waiting on a stopped state")
	}
}

func Test_newReadyCancelPending(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	ready.setError(nil)
	ctx, cancel := context.WithCancel(context.Background())
	// create 10 goroutines stuck on pending
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait(ctx)
		}()
	}
	select {
	case <-time.After(1 * time.Second):
	case <-errCh:
		t.Errorf("ready should be blocking")
	}
	cancel()
	for i := 0; i < 10; i++ {
		if err := <-errCh; err == nil {
			t.Errorf("unexpected success on channel %d", i)
		}
	}
}

func Test_newReadyStateChangeTimestamp(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	fakeClock.SetTime(time.Now())

	ready := newReady(fakeClock)
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, time.Minute)

	ready.setReady()
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, time.Minute)
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, 2*time.Minute)

	ready.setError(nil)
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, time.Minute)
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, 2*time.Minute)

	ready.setReady()
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, time.Minute)

	ready.stop()
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, time.Minute)
	fakeClock.Step(time.Minute)
	checkReadyTransitionTime(t, ready, 2*time.Minute)
}

func checkReadyTransitionTime(t *testing.T, r *ready, expectedLastStateChangeDuration time.Duration) {
	if lastStateChangeDuration, _ := r.check(); lastStateChangeDuration != expectedLastStateChangeDuration {
		t.Errorf("unexpected last state change duration: %v, expected: %v", lastStateChangeDuration, expectedLastStateChangeDuration)
	}
}

func TestReadyError(t *testing.T) {
	ready := newReady(testingclock.NewFakeClock(time.Now()))
	_, _, err := ready.checkAndReadGeneration()
	if err == nil || err.Error() != "storage is (re)initializing" {
		t.Errorf("Unexpected error when unready, got %q", err)
	}
	ready.setError(errors.New("etcd is down"))
	_, _, err = ready.checkAndReadGeneration()
	if err == nil || err.Error() != "storage is (re)initializing: etcd is down" {
		t.Errorf("Unexpected error when unready, got %q", err)
	}
	ready.setError(nil)
	_, _, err = ready.checkAndReadGeneration()
	if err == nil || err.Error() != "storage is (re)initializing" {
		t.Errorf("Unexpected error when unready, got %q", err)
	}
}
