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

package runtime

import (
	"sync"
)

// MockMetricsRecorder is a fake implementation of MetricsRecorder for testing.
// It uses counters and stores call records for verification in tests.
type MockMetricsRecorder struct {
	mu sync.Mutex

	// Counters for method calls
	pluginDurationCalls int
	flushMetricsCalls   int

	// Records of calls for verification
	pluginDurationRecords []PluginDurationRecord
}

// PluginDurationRecord stores the parameters of ObservePluginDurationAsync calls
type PluginDurationRecord struct {
	ExtensionPoint string
	PluginName     string
	Status         string
	Value          float64
}

// Queueing hint and in-flight event metrics are intentionally not
// recorded in this runtime mock. The methods exist as no-ops to satisfy
// the interface; backend-specific tests should provide their own mocks
// if they need to assert on those metrics.

// NewMockMetricsRecorder creates a new MockMetricsRecorder
func NewMockMetricsRecorder() *MockMetricsRecorder {
	return &MockMetricsRecorder{
		pluginDurationRecords: []PluginDurationRecord{},
	}
}

// ObservePluginDurationAsync records the plugin duration observation
func (f *MockMetricsRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls++
	f.pluginDurationRecords = append(f.pluginDurationRecords, PluginDurationRecord{
		ExtensionPoint: extensionPoint,
		PluginName:     pluginName,
		Status:         status,
		Value:          value,
	})
}

// ObserveQueueingHintDurationAsync is a no-op in the runtime mock.
func (f *MockMetricsRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
}

// ObserveInFlightEventsAsync is a no-op in the runtime mock.
func (f *MockMetricsRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
}

// FlushMetrics records the flush operation
func (f *MockMetricsRecorder) FlushMetrics() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.flushMetricsCalls++
}

// PluginDurationCallCount returns the number of ObservePluginDurationAsync calls
func (f *MockMetricsRecorder) PluginDurationCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pluginDurationCalls
}

// FlushMetricsCallCount returns the number of FlushMetrics calls
func (f *MockMetricsRecorder) FlushMetricsCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.flushMetricsCalls
}

// GetPluginDurationRecords returns a copy of the plugin duration records
func (f *MockMetricsRecorder) GetPluginDurationRecords() []PluginDurationRecord {
	f.mu.Lock()
	defer f.mu.Unlock()

	records := make([]PluginDurationRecord, len(f.pluginDurationRecords))
	copy(records, f.pluginDurationRecords)
	return records
}

// Get queueing hint and in-flight event records are intentionally absent.

// Reset clears all counters and records
func (f *MockMetricsRecorder) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls = 0
	f.flushMetricsCalls = 0

	f.pluginDurationRecords = []PluginDurationRecord{}
}
