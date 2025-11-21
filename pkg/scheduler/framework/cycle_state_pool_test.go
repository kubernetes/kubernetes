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

package framework

import (
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// TestCycleStatePool tests that the cycleStatePool correctly reuses objects
func TestCycleStatePool(t *testing.T) {
	// Get a CycleState from the pool
	cs1 := NewCycleState()
	if cs1 == nil {
		t.Fatal("Expected non-nil CycleState from NewCycleState")
	}
	
	// Use the CycleState
	cs1.SetRecordPluginMetrics(true)
	cs1.Write("test-key", &testData{value: "test-value"})
	
	// Recycle it
	cs1.Recycle()
	
	// Get another CycleState
	cs2 := NewCycleState()
	
	// It should be clean
	if cs2.ShouldRecordPluginMetrics() {
		t.Error("Expected recordPluginMetrics to be false after getting from pool")
	}
	
	// Check that storage is empty
	if _, err := cs2.Read("test-key"); err != fwk.ErrNotFound {
		t.Errorf("Expected ErrNotFound for 'test-key', got %v", err)
	}
	
	// Clean up
	cs2.Recycle()
}

// TestCycleStatePoolConcurrent tests concurrent access to the CycleState pool
func TestCycleStatePoolConcurrent(t *testing.T) {
	const numGoroutines = 50
	const numIterations = 100
	
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numIterations)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numIterations; j++ {
				// Get CycleState from pool
				cs := NewCycleState()
				if cs == nil {
					errors <- testError{"Got nil CycleState"}
					return
				}
				
				// Use it
				key := fwk.StateKey("key-" + string(rune(id)))
				data := &testData{value: "value"}
				cs.Write(key, data)
				
				// Read back and verify
				readData, err := cs.Read(key)
				if err != nil {
					errors <- testError{"Failed to read back data: " + err.Error()}
					return
				}
				if readData.(*testData).value != data.value {
					errors <- testError{"Data mismatch"}
					return
				}
				
				// Set some skip plugins
				cs.SetSkipFilterPlugins(sets.New("plugin1", "plugin2"))
				cs.SetSkipScorePlugins(sets.New("plugin3"))
				
				// Recycle
				cs.Recycle()
			}
		}(i)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for errors
	for err := range errors {
		t.Error(err)
	}
}

// TestCycleStateRecycle tests that Recycle properly cleans the CycleState
func TestCycleStateRecycle(t *testing.T) {
	cs := NewCycleState()
	
	// Add some data
	cs.Write("key1", &testData{value: "value1"})
	cs.Write("key2", &testData{value: "value2"})
	cs.SetRecordPluginMetrics(true)
	cs.SetSkipFilterPlugins(sets.New("plugin1"))
	cs.SetSkipScorePlugins(sets.New("plugin2"))
	cs.SetSkipPreBindPlugins(sets.New("plugin3"))
	
	// Recycle
	cs.Recycle()
	
	// Get a new one (should be the recycled one)
	cs2 := NewCycleState()
	
	// Verify it's clean
	if _, err := cs2.Read("key1"); err != fwk.ErrNotFound {
		t.Errorf("Expected key1 to be cleared, got %v", err)
	}
	if _, err := cs2.Read("key2"); err != fwk.ErrNotFound {
		t.Errorf("Expected key2 to be cleared, got %v", err)
	}
	if cs2.ShouldRecordPluginMetrics() {
		t.Error("Expected recordPluginMetrics to be false")
	}
	if cs2.GetSkipFilterPlugins() != nil {
		t.Error("Expected skipFilterPlugins to be nil")
	}
	if cs2.GetSkipScorePlugins() != nil {
		t.Error("Expected skipScorePlugins to be nil")
	}
	if cs2.GetSkipPreBindPlugins() != nil {
		t.Error("Expected skipPreBindPlugins to be nil")
	}
	
	cs2.Recycle()
}

// TestCycleStateRecycleNil tests that Recycle handles nil properly
func TestCycleStateRecycleNil(t *testing.T) {
	var cs *CycleState
	// This should not panic
	cs.Recycle()
}

// testData implements StateData for testing
type testData struct {
	value string
}

func (t *testData) Clone() fwk.StateData {
	return &testData{value: t.value}
}

// testError is a simple error type for testing
type testError struct {
	msg string
}

func (e testError) Error() string {
	return e.msg
}
