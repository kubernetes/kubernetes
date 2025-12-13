/*
Copyright 2024 The Kubernetes Authors.

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

package scheduler

import (
	"sync"
	"testing"

	fwk "k8s.io/kube-scheduler/framework"
)

// TestNodeStatusPool tests that the nodeStatusPool correctly reuses objects
func TestNodeStatusPool(t *testing.T) {
	// Get an object from the pool
	ns1 := nodeStatusPool.Get().(*nodeStatus)
	if ns1 == nil {
		t.Fatal("Expected non-nil nodeStatus from pool")
	}
	
	// Set some values
	ns1.node = "test-node-1"
	ns1.status = fwk.NewStatus(fwk.Success)
	
	// Store the pointer for later comparison
	ptr1 := ns1
	
	// Reset and return to pool
	ns1.node = ""
	ns1.status = nil
	nodeStatusPool.Put(ns1)
	
	// Get another object from the pool
	ns2 := nodeStatusPool.Get().(*nodeStatus)
	
	// Should get the same object back (pointer equality)
	if ptr1 != ns2 {
		t.Log("Note: Got different object from pool, which is valid but not optimal for reuse")
	}
	
	// Ensure the object is clean
	if ns2.node != "" {
		t.Errorf("Expected empty node field, got %s", ns2.node)
	}
	if ns2.status != nil {
		t.Errorf("Expected nil status field, got %v", ns2.status)
	}
}

// TestNodeStatusPoolConcurrent tests concurrent access to the nodeStatusPool
func TestNodeStatusPoolConcurrent(t *testing.T) {
	const numGoroutines = 100
	const numIterations = 1000
	
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numIterations; j++ {
				// Get from pool
				ns := nodeStatusPool.Get().(*nodeStatus)
				if ns == nil {
					errors <- &testError{msg: "Got nil from pool"}
					return
				}
				
				// Use the object
				ns.node = "node-test"
				ns.status = fwk.NewStatus(fwk.Success)
				
				// Verify we can read back what we wrote
				if ns.node != "node-test" {
					errors <- &testError{msg: "Node field corrupted"}
					return
				}
				
				// Clean and return to pool
				ns.node = ""
				ns.status = nil
				nodeStatusPool.Put(ns)
			}
		}(i)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for any errors
	for err := range errors {
		t.Error(err)
	}
}

// TestNodeStatusPoolReset tests that objects are properly reset before returning to pool
func TestNodeStatusPoolReset(t *testing.T) {
	// Test that we handle nil status properly
	ns := nodeStatusPool.Get().(*nodeStatus)
	ns.node = "test-node"
	ns.status = nil // Explicitly set to nil
	
	// This should not panic
	ns.node = ""
	nodeStatusPool.Put(ns)
	
	// Get it back and verify it's clean
	ns2 := nodeStatusPool.Get().(*nodeStatus)
	if ns2.node != "" || ns2.status != nil {
		t.Errorf("Object not properly reset: node=%s, status=%v", ns2.node, ns2.status)
	}
}

// testError is a simple error type for testing
type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}
