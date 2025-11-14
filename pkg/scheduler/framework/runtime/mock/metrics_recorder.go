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

package mock

import (
	"slices"
	"sync"
)

// MetricsRecorder is a fake implementation of MetricsRecorder for testing.
// It uses counters and stores call records for verification in tests.
// Queueing hint and in-flight event metrics are intentionally not
// recorded in this runtime mock. The methods exist as no-ops to satisfy
// the interface; backend-specific tests should provide their own mocks
// if they need to assert on those metrics.
// Get queueing hint and in-flight event records are intentionally absent.
type MetricsRecorder struct {
	mu sync.Mutex

	// Counters for method calls
	pluginDurationCalls int

	// pluginDurationRecords is used to store the records of calls and is used for verification
	pluginDurationRecords []PluginDurationRecord
}

// PluginDurationRecord stores the parameters of ObservePluginDurationAsync calls
type PluginDurationRecord struct {
	ExtensionPoint string
	PluginName     string
	Status         string
	Value          float64
}

// NewMetricsRecorder creates a new MockMetricsRecorder
func NewMetricsRecorder() *MetricsRecorder {
	return &MetricsRecorder{}
}

// ObservePluginDurationAsync records the plugin duration observation
func (f *MetricsRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
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
func (f *MetricsRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
}

// ObserveInFlightEventsAsync is a no-op in the runtime mock.
func (f *MetricsRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
}

// FlushMetrics is a no-op in the runtime mock.
func (f *MetricsRecorder) FlushMetrics() {
}

// PluginDurationCallCount returns the number of ObservePluginDurationAsync calls
func (f *MetricsRecorder) PluginDurationCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.pluginDurationRecords)
}

// FlushMetricsCallCount is a no-op in the runtime mock.
func (f *MetricsRecorder) FlushMetricsCallCount() int {
	return 0
}

// GetPluginDurationRecords returns a copy of the plugin duration records
func (f *MetricsRecorder) GetPluginDurationRecords() []PluginDurationRecord {
	f.mu.Lock()
	defer f.mu.Unlock()

	return slices.Clone(f.pluginDurationRecords)
}

// Reset clears all counters and records
func (f *MetricsRecorder) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls = 0
	f.pluginDurationRecords = []PluginDurationRecord{}
}
