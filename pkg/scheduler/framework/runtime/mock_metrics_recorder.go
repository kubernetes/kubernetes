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
	queueingHintCalls   int
	inFlightEventsCalls int
	flushMetricsCalls   int

	// Records of calls for verification
	pluginDurationRecords []PluginDurationRecord
	queueingHintRecords   []QueueingHintRecord
	inFlightEventsRecords []InFlightEventsRecord
}

// PluginDurationRecord stores the parameters of ObservePluginDurationAsync calls
type PluginDurationRecord struct {
	ExtensionPoint string
	PluginName     string
	Status         string
	Value          float64
}

// QueueingHintRecord stores the parameters of ObserveQueueingHintDurationAsync calls
type QueueingHintRecord struct {
	PluginName string
	Event      string
	Hint       string
	Value      float64
}

// InFlightEventsRecord stores the parameters of ObserveInFlightEventsAsync calls
type InFlightEventsRecord struct {
	EventLabel string
	ValueToAdd float64
	ForceFlush bool
}

// NewMockMetricsRecorder creates a new MockMetricsRecorder
func NewMockMetricsRecorder() *MockMetricsRecorder {
	return &MockMetricsRecorder{
		pluginDurationRecords: []PluginDurationRecord{},
		queueingHintRecords:   []QueueingHintRecord{},
		inFlightEventsRecords: []InFlightEventsRecord{},
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

// ObserveQueueingHintDurationAsync records the queueing hint duration observation
func (f *MockMetricsRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.queueingHintCalls++
	f.queueingHintRecords = append(f.queueingHintRecords, QueueingHintRecord{
		PluginName: pluginName,
		Event:      event,
		Hint:       hint,
		Value:      value,
	})
}

// ObserveInFlightEventsAsync records the in-flight events observation
func (f *MockMetricsRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.inFlightEventsCalls++
	f.inFlightEventsRecords = append(f.inFlightEventsRecords, InFlightEventsRecord{
		EventLabel: eventLabel,
		ValueToAdd: valueToAdd,
		ForceFlush: forceFlush,
	})
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

// QueueingHintCallCount returns the number of ObserveQueueingHintDurationAsync calls
func (f *MockMetricsRecorder) QueueingHintCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.queueingHintCalls
}

// InFlightEventsCallCount returns the number of ObserveInFlightEventsAsync calls
func (f *MockMetricsRecorder) InFlightEventsCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.inFlightEventsCalls
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

// GetQueueingHintRecords returns a copy of the queueing hint records
func (f *MockMetricsRecorder) GetQueueingHintRecords() []QueueingHintRecord {
	f.mu.Lock()
	defer f.mu.Unlock()

	records := make([]QueueingHintRecord, len(f.queueingHintRecords))
	copy(records, f.queueingHintRecords)
	return records
}

// GetInFlightEventsRecords returns a copy of the in-flight events records
func (f *MockMetricsRecorder) GetInFlightEventsRecords() []InFlightEventsRecord {
	f.mu.Lock()
	defer f.mu.Unlock()

	records := make([]InFlightEventsRecord, len(f.inFlightEventsRecords))
	copy(records, f.inFlightEventsRecords)
	return records
}

// Reset clears all counters and records
func (f *MockMetricsRecorder) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls = 0
	f.queueingHintCalls = 0
	f.inFlightEventsCalls = 0
	f.flushMetricsCalls = 0

	f.pluginDurationRecords = []PluginDurationRecord{}
	f.queueingHintRecords = []QueueingHintRecord{}
	f.inFlightEventsRecords = []InFlightEventsRecord{}
}
