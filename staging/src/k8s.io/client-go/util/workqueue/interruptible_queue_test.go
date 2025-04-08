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

package workqueue_test

import (
	"context"
	"testing"
	"time"

	"k8s.io/client-go/util/workqueue"
)

func TestInterruptibleQueueBasic(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Add item to queue
	q.Add("item1")

	// Get item using normal Get
	item, shutdown := q.Get()
	if shutdown {
		t.Errorf("Expected queue to not be shutdown")
	}
	if item != "item1" {
		t.Errorf("Expected %v, got %v", "item1", item)
	}

	// Done with processing
	q.Done(item)

	// Test with GetWithContext
	q.Add("item2")
	ctx := context.Background()
	item, shutdown, err := q.GetWithContext(ctx)
	if err != nil {
		t.Errorf("Did not expect error: %v", err)
	}
	if shutdown {
		t.Errorf("Expected queue to not be shutdown")
	}
	if item != "item2" {
		t.Errorf("Expected %v, got %v", "item2", item)
	}
	q.Done(item)

	// Queue should be empty now
	if q.Len() != 0 {
		t.Errorf("Expected queue to be empty. Has %v items", q.Len())
	}

	// Shutdown the queue
	q.ShutDown()
	_, shutdown = q.Get()
	if !shutdown {
		t.Errorf("Expected queue to be shutdown")
	}

	// ShutDown queue should return shutdown = true with GetWithContext
	_, shutdown, _ = q.GetWithContext(ctx)
	if !shutdown {
		t.Errorf("Expected queue to be shutdown with GetWithContext")
	}
}

func TestInterruptibleQueueWithContextCancellation(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Create a context that will be cancelled
	ctx, cancel := context.WithCancel(context.Background())

	// Start a goroutine that will cancel the context after a short delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	// Try to get from empty queue with context that will be cancelled
	start := time.Now()
	_, shutdown, err := q.GetWithContext(ctx)
	elapsed := time.Since(start)

	// Verify we got cancelled and didn't wait too long
	if shutdown {
		t.Errorf("Expected queue to not be shutdown")
	}
	if err == nil {
		t.Errorf("Expected error from context cancellation")
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("Get took too long: %v", elapsed)
	}
}

func TestInterruptibleQueueAddWhileWaiting(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Create a context with long timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Start a goroutine that will add an item after a short delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		q.Add("delayed-item")
	}()

	// Try to get from empty queue, should wait until item is added
	start := time.Now()
	item, shutdown, err := q.GetWithContext(ctx)
	elapsed := time.Since(start)

	// Verify we got the item and didn't return immediately
	if err != nil {
		t.Errorf("Did not expect error: %v", err)
	}
	if shutdown {
		t.Errorf("Expected queue to not be shutdown")
	}
	if item != "delayed-item" {
		t.Errorf("Expected %v, got %v", "delayed-item", item)
	}
	if elapsed < 40*time.Millisecond {
		t.Errorf("Get returned too quickly: %v", elapsed)
	}

	q.Done(item)
}

func TestInterruptibleQueueShutdownWhileWaiting(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Create a context with long timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Start a goroutine that will shutdown the queue after a short delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		q.ShutDown()
	}()

	// Try to get from empty queue, should return shutdown=true when queue is shutdown
	start := time.Now()
	_, shutdown, err := q.GetWithContext(ctx)
	elapsed := time.Since(start)

	// Verify we detected shutdown
	if err != nil {
		t.Errorf("Did not expect error: %v", err)
	}
	if !shutdown {
		t.Errorf("Expected queue to be shutdown")
	}
	if elapsed < 40*time.Millisecond {
		t.Errorf("Get returned too quickly: %v", elapsed)
	}
}

func TestInterruptibleQueueCancelledContext(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Create already cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Try to get with already cancelled context
	_, shutdown, err := q.GetWithContext(ctx)

	// Verify we get error without being shutdown
	if shutdown {
		t.Errorf("Expected queue to not be shutdown")
	}
	if err == nil {
		t.Errorf("Expected context cancellation error")
	}
}

func TestInterruptibleQueueWithDrain(t *testing.T) {
	q := workqueue.NewTypedInterruptibleQueue[string]()

	// Add items
	q.Add("item1")
	q.Add("item2")

	// Get one item
	item, _ := q.Get()
	if item != "item1" {
		t.Errorf("Expected %v, got %v", "item1", item)
	}

	// Must mark the processing item as done before calling ShutDownWithDrain
	q.Done(item)

	// Shutdown with drain
	q.ShutDownWithDrain()

	// Try to add after shutdown
	q.Add("item3")

	// Should still be able to get remaining item
	ctx := context.Background()
	item, shutdown, err := q.GetWithContext(ctx)
	if err != nil {
		t.Errorf("Did not expect error: %v", err)
	}
	// Note: At this point the queue is marked as shuttingDown, but since there are still items in the queue, the shutdown flag is false
	if shutdown {
		t.Errorf("Expected queue to not return shutdown=true as there are still items in queue")
	}
	if item != "item2" {
		t.Errorf("Expected %v, got %v", "item2", item)
	}

	// Complete processing the last item
	q.Done(item)

	// Queue should now return empty results with shutdown=true
	_, shutdown, _ = q.GetWithContext(ctx)
	if !shutdown {
		t.Errorf("Expected queue to be shutdown")
	}
}
