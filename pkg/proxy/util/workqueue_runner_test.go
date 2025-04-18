/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"sync"
	"testing"
	"time"
)

// TestWorkQueueRunnerBasic tests the basic functionality of WorkQueueRunner
func TestWorkQueueRunnerBasic(t *testing.T) {
	var callCount int
	var callLock sync.Mutex
	
	fn := func() {
		callLock.Lock()
		defer callLock.Unlock()
		callCount++
	}
	
	minInterval := 100 * time.Millisecond
	maxInterval := 500 * time.Millisecond
	
	runner := NewWorkQueueRunner("test-runner", fn, minInterval, maxInterval)
	
	stop := make(chan struct{})
	defer close(stop)
	
	go runner.Loop(stop)
	
	// Wait for the first run to happen (should be immediate)
	time.Sleep(50 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 1 {
		t.Errorf("Expected 1 call, got %d", callCount)
	}
	callLock.Unlock()
	
	// Run again, should be throttled
	runner.Run()
	time.Sleep(50 * time.Millisecond) // Not enough time has passed
	
	callLock.Lock()
	if callCount != 1 {
		t.Errorf("Expected still 1 call (throttled), got %d", callCount)
	}
	callLock.Unlock()
	
	// Wait for minInterval to pass, should run again
	time.Sleep(100 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 2 {
		t.Errorf("Expected 2 calls after minInterval, got %d", callCount)
	}
	callLock.Unlock()
	
	// Wait for maxInterval to pass, should run again even without explicit Run() call
	time.Sleep(500 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 3 {
		t.Errorf("Expected 3 calls after maxInterval, got %d", callCount)
	}
	callLock.Unlock()
}

// TestWorkQueueRunnerRetryAfter tests the RetryAfter functionality
func TestWorkQueueRunnerRetryAfter(t *testing.T) {
	var callCount int
	var callLock sync.Mutex
	var shouldRetry bool
	
	fn := func() {
		callLock.Lock()
		defer callLock.Unlock()
		callCount++
		
		if shouldRetry {
			shouldRetry = false
			// Schedule a retry after a short interval
			go func() {
				time.Sleep(10 * time.Millisecond)
				runner.RetryAfter(50 * time.Millisecond)
			}()
		}
	}
	
	minInterval := 200 * time.Millisecond
	maxInterval := 1000 * time.Millisecond
	
	runner := NewWorkQueueRunner("test-runner", fn, minInterval, maxInterval)
	
	stop := make(chan struct{})
	defer close(stop)
	
	go runner.Loop(stop)
	
	// Wait for the first run to happen
	time.Sleep(50 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 1 {
		t.Errorf("Expected 1 call, got %d", callCount)
	}
	callLock.Unlock()
	
	// Set up retry and run again
	callLock.Lock()
	shouldRetry = true
	callLock.Unlock()
	runner.Run()
	
	// Wait for minInterval to pass for the second run
	time.Sleep(200 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 2 {
		t.Errorf("Expected 2 calls after minInterval, got %d", callCount)
	}
	callLock.Unlock()
	
	// Wait for the retry to happen (50ms after the second run)
	time.Sleep(100 * time.Millisecond)
	
	callLock.Lock()
	if callCount != 3 {
		t.Errorf("Expected 3 calls after retry, got %d", callCount)
	}
	callLock.Unlock()
}
