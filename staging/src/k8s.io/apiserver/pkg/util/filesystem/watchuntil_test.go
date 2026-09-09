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

package filesystem

import (
	"context"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestWatchUntilPanicsOnInvalidArgs(t *testing.T) {
	t.Run("zero pollInterval panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for zero pollInterval")
			}
		}()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		WatchUntil(ctx, 0, "/tmp", func() {}, nil)
	})

	t.Run("negative pollInterval panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for negative pollInterval")
			}
		}()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		WatchUntil(ctx, -1*time.Second, "/tmp", func() {}, nil)
	})

	t.Run("nil eventHandler panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for nil eventHandler")
			}
		}()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		WatchUntil(ctx, time.Second, "/tmp", nil, nil)
	})
}

func TestWatchUntilContextCancellation(t *testing.T) {
	dir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	var callCount atomic.Int32

	done := make(chan struct{})
	go func() {
		defer close(done)
		WatchUntil(ctx, 50*time.Millisecond, dir, func() {
			callCount.Add(1)
		}, nil)
	}()

	// Wait for at least one handler invocation
	waitForCondition(t, 2*time.Second, func() bool {
		return callCount.Load() > 0
	})

	cancel()

	select {
	case <-done:
		// WatchUntil returned after context cancellation
	case <-time.After(5 * time.Second):
		t.Fatal("WatchUntil did not return after context cancellation")
	}
}

func TestWatchUntilCallsHandlerOnPoll(t *testing.T) {
	dir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var callCount atomic.Int32

	go func() {
		WatchUntil(ctx, 50*time.Millisecond, dir, func() {
			callCount.Add(1)
		}, nil)
	}()

	// Should get called multiple times via polling
	waitForCondition(t, 2*time.Second, func() bool {
		return callCount.Load() >= 3
	})
}

func TestWatchUntilCallsHandlerOnFileChange(t *testing.T) {
	dir := t.TempDir()
	testFile := filepath.Join(dir, "test.yaml")
	if err := os.WriteFile(testFile, []byte("initial"), 0644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var callCount atomic.Int32

	go func() {
		WatchUntil(ctx, 10*time.Minute, dir, func() {
			callCount.Add(1)
		}, nil)
	}()

	// Wait for initial handler call
	waitForCondition(t, 2*time.Second, func() bool {
		return callCount.Load() >= 1
	})

	initialCount := callCount.Load()

	// Modify the file to trigger a filesystem event
	if err := os.WriteFile(testFile, []byte("modified"), 0644); err != nil {
		t.Fatal(err)
	}

	// Should get called again due to file change
	waitForCondition(t, 5*time.Second, func() bool {
		return callCount.Load() > initialCount
	})
}

func TestWatchUntilNonExistentPathCallsErrorHandler(t *testing.T) {
	nonExistent := filepath.Join(t.TempDir(), "does-not-exist")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var errorCount atomic.Int32
	var eventCount atomic.Int32

	go func() {
		WatchUntil(ctx, 50*time.Millisecond, nonExistent, func() {
			eventCount.Add(1)
		}, func(err error) {
			errorCount.Add(1)
		})
	}()

	// Error handler should be called for the non-existent path
	waitForCondition(t, 2*time.Second, func() bool {
		return errorCount.Load() > 0
	})

	// Event handler should still be called via polling
	waitForCondition(t, 2*time.Second, func() bool {
		return eventCount.Load() > 0
	})
}

func TestWatchUntilNilErrorHandler(t *testing.T) {
	nonExistent := filepath.Join(t.TempDir(), "does-not-exist")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var eventCount atomic.Int32

	// nil errorHandler should not panic
	done := make(chan struct{})
	go func() {
		defer close(done)
		WatchUntil(ctx, 50*time.Millisecond, nonExistent, func() {
			eventCount.Add(1)
		}, nil)
	}()

	// Should still work with polling
	waitForCondition(t, 2*time.Second, func() bool {
		return eventCount.Load() > 0
	})

	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("WatchUntil did not return")
	}
}

func TestWatchUntilDetectsAtomicRename(t *testing.T) {
	dir := t.TempDir()
	testFile := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(testFile, []byte("version: 1"), 0644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var callCount atomic.Int32

	go func() {
		// Use a long poll interval so we rely on fsnotify, not polling
		WatchUntil(ctx, 10*time.Minute, dir, func() {
			callCount.Add(1)
		}, nil)
	}()

	// Wait for initial handler call
	waitForCondition(t, 2*time.Second, func() bool {
		return callCount.Load() >= 1
	})

	countBefore := callCount.Load()

	// Atomic file replacement: write temp, then rename (recommended KEP pattern)
	tmpFile := filepath.Join(dir, ".config.yaml.tmp")
	if err := os.WriteFile(tmpFile, []byte("version: 2"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmpFile, testFile); err != nil {
		t.Fatal(err)
	}

	// Should detect the rename event
	waitForCondition(t, 5*time.Second, func() bool {
		return callCount.Load() > countBefore
	})
}

func waitForCondition(t *testing.T, timeout time.Duration, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("condition not met within %v", timeout)
}
