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
	"sync/atomic"
	"testing"
	"time"
)

func TestNewBoundedFrequencyRunner(t *testing.T) {
	testFn := func() {}
	minInterval := 100 * time.Millisecond
	maxInterval := 500 * time.Millisecond
	name := "test-runner"
	burst := 3

	runner := NewBoundedFrequencyRunner(name, testFn, minInterval, maxInterval, burst)

	if runner.maxInterval != maxInterval {
		t.Errorf("NewBoundedFrequencyRunner: maxInterval mismatch, got %v, want %v", runner.maxInterval, maxInterval)
	}

	if runner.fn == nil {
		t.Errorf("NewBoundedFrequencyRunner: fn is nil, want not nil")
	}
	if runner.queue == nil {
		t.Errorf("NewBoundedFrequencyRunner: queue is nil, want not nil")
	}

	// Test panic condition
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("NewBoundedFrequencyRunner: did not panic with invalid intervals")
		}
	}()
	NewBoundedFrequencyRunner(name, testFn, maxInterval, minInterval, burst)
}

func TestBoundedFrequencyRunner(t *testing.T) {
	tests := []struct {
		name             string
		minInterval      time.Duration
		maxInterval      time.Duration
		burst            int
		runFunc          func(runner *BoundedFrequencyRunner) // Define a function type for running
		sleepDuration    time.Duration
		expectedRunCount int
	}{
		{
			name:        "Test Run",
			minInterval: 100 * time.Millisecond,
			maxInterval: 500 * time.Millisecond,
			burst:       1,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.Run()
				runner.Run()
				runner.Run()
			},
			sleepDuration:    200 * time.Millisecond,
			expectedRunCount: 1,
		},
		{
			name:        "Test Run with longer sleep",
			minInterval: 10 * time.Millisecond,
			maxInterval: 200 * time.Millisecond,
			burst:       1,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.Run()
			},
			sleepDuration:    300 * time.Millisecond,
			expectedRunCount: 2, // Should run twice because of maxInterval
		},
		{
			name:        "Test RetryAfter",
			minInterval: 50 * time.Millisecond,
			maxInterval: 500 * time.Millisecond,
			burst:       1,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.RetryAfter(100 * time.Millisecond)
			},
			sleepDuration:    300 * time.Millisecond,
			expectedRunCount: 1,
		},
		{
			name:        "Test Burst",
			minInterval: 50 * time.Millisecond,
			maxInterval: 500 * time.Millisecond,
			burst:       3,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
			},
			sleepDuration:    15 * time.Millisecond,
			expectedRunCount: 3,
		},
		{
			name:        "Test Burst and minInterval",
			minInterval: 50 * time.Millisecond,
			maxInterval: 200 * time.Millisecond,
			burst:       2,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
			},
			sleepDuration:    20 * time.Millisecond, // Sleep more than minInterval
			expectedRunCount: 2,
		},
		{
			name:        "Test Burst and MaxInterval",
			minInterval: 50 * time.Millisecond,
			maxInterval: 200 * time.Millisecond,
			burst:       2,
			runFunc: func(runner *BoundedFrequencyRunner) {
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
				time.Sleep(5 * time.Millisecond)
				runner.Run()
			},
			sleepDuration:    230 * time.Millisecond,
			expectedRunCount: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var runCount atomic.Int32
			testFn := func() {
				runCount.Add(1)
			}

			runner := NewBoundedFrequencyRunner("test-runner", testFn, tt.minInterval, tt.maxInterval, tt.burst)
			stopChan := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(1)

			go func() {
				defer wg.Done()
				runner.Loop(stopChan)
			}()

			// Run the test function
			tt.runFunc(runner)

			// Wait for the specified duration
			time.Sleep(tt.sleepDuration)
			close(stopChan) // Signal the loop to stop
			wg.Wait()       // Wait for the loop to finish

			if int(runCount.Load()) != tt.expectedRunCount {
				t.Errorf("%s: expected %d runs, got %d", tt.name, tt.expectedRunCount, runCount.Load())
			}
		})
	}
}

func TestBoundedFrequencyRunnerRun(t *testing.T) {
	var executionCount atomic.Int32
	var lastExecutionTime time.Time
	minInterval := 100 * time.Millisecond
	maxInterval := 500 * time.Millisecond
	burst := 1

	runner := NewBoundedFrequencyRunner("test", func() {
		executionCount.Add(1)
		now := time.Now()
		if !lastExecutionTime.IsZero() {
			if now.Before(lastExecutionTime.Add(minInterval)) {
				t.Errorf("function executed too soon: last execution at %v, current at %v, min interval %v", lastExecutionTime, now, minInterval)
			}
		}
		lastExecutionTime = now
	}, minInterval, maxInterval, burst)

	stop := make(chan struct{})
	defer close(stop)
	go runner.Loop(stop)
	time.Sleep(10 * time.Millisecond) // Give the loop a chance to start

	// Trigger multiple runs quickly, they should be coalesced
	runner.Run()
	runner.Run()
	runner.Run()

	// Wait for at least one execution
	time.Sleep(2 * minInterval)
	if executionCount.Load() < 1 {
		t.Errorf("function should have executed at least once")
	}

	initialCount := int(executionCount.Load())
	time.Sleep(2 * minInterval)
	if int(executionCount.Load()) > initialCount+burst {
		t.Errorf("should not have executed more than burst times in this interval, got %d, expected max %d", int(executionCount.Load())-initialCount, burst)
	}
}

func TestBoundedFrequencyRunnerRetryAfter(t *testing.T) {
	var executionCount atomic.Int32
	delay := 200 * time.Millisecond
	minInterval := 100 * time.Millisecond
	maxInterval := 500 * time.Millisecond
	burst := 1

	runner := NewBoundedFrequencyRunner("test", func() {
		executionCount.Add(1)
	}, minInterval, maxInterval, burst)

	stop := make(chan struct{})
	defer close(stop)
	go runner.Loop(stop)
	time.Sleep(10 * time.Millisecond) // Give the loop a chance to start

	startTime := time.Now()
	runner.RetryAfter(delay)

	// Wait for slightly longer than the delay
	time.Sleep(delay + 50*time.Millisecond)

	if executionCount.Load() != 1 {
		t.Errorf("function should have executed exactly once after RetryAfter, got %d", executionCount.Load())
	}

	endTime := time.Now()
	if endTime.Before(startTime.Add(delay)) {
		t.Errorf("function executed too early after RetryAfter, expected at least %v, got %v", delay, endTime.Sub(startTime))
	}
}

func TestBoundedFrequencyRunnerBurst(t *testing.T) {
	var executionCount atomic.Int32
	minInterval := 100 * time.Millisecond
	maxInterval := time.Minute
	burst := 3

	runner := NewBoundedFrequencyRunner("test", func() {
		executionCount.Add(1)
	}, minInterval, maxInterval, burst)

	stop := make(chan struct{})
	defer close(stop)
	go runner.Loop(stop)
	time.Sleep(10 * time.Millisecond) // Give the loop a chance to start

	// Trigger more runs than the burst
	start := time.Now()
	for i := 0; i < burst+5; i++ {
		runner.Run()
		// wait for the function to finish
		time.Sleep(5 * time.Millisecond)
	}

	if time.Since(start) > minInterval {
		t.Fatalf("burst functions should be executed within an interval, took %v", time.Since(start))
	}

	time.Sleep(minInterval)
	// We expect the function to have executed up to 'burst' times relatively quickly
	// and then throttled.
	if int(executionCount.Load()) != burst {
		t.Errorf("expected %d executions within the burst window, got %d", burst, executionCount.Load())
	}

}

func TestBoundedFrequencyRunnerNoRunCalls(t *testing.T) {
	var executionCount atomic.Int32
	minInterval := 10 * time.Millisecond
	maxInterval := 100 * time.Millisecond
	burst := 1

	runner := NewBoundedFrequencyRunner("test", func() {
		executionCount.Add(1)
	}, minInterval, maxInterval, burst)

	stop := make(chan struct{})
	defer close(stop)
	go runner.Loop(stop)
	time.Sleep(10 * time.Millisecond) // Give the loop a chance to start

	// Don't call Run at all, rely solely on maxInterval
	time.Sleep(8 * maxInterval)

	if executionCount.Load() < 3 {
		t.Errorf("function should have executed at least three times based on maxInterval, got %d", executionCount.Load())
	}
}
