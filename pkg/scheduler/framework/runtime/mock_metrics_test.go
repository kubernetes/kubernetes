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

package runtime

import (
	"sync"
)

// metricRecorderInterface defines the interface that both real and mock recorders implement
type metricRecorderInterface interface {
	ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64)
	ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64)
	ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool)
	FlushMetrics()
}

// mockMetricAsyncRecorder is a mock implementation of metrics.MetricAsyncRecorder
// for testing purposes. It records all method calls for verification.
type mockMetricAsyncRecorder struct {
	mu sync.Mutex
	// observePluginDurationCalls records calls to ObservePluginDurationAsync
	observePluginDurationCalls []pluginDurationCall
	// observeQueueingHintCalls records calls to ObserveQueueingHintDurationAsync
	observeQueueingHintCalls []queueingHintCall
	// inflightEventCalls records calls to ObserveInFlightEventsAsync
	inflightEventCalls []inflightEventCall
	// flushed indicates if FlushMetrics was called
	flushed bool

	// IsStoppedCh mimics the real MetricAsyncRecorder for compatibility
	IsStoppedCh chan struct{}
}

type pluginDurationCall struct {
	extensionPoint string
	plugin         string
	status         string
	duration       float64
}

type queueingHintCall struct {
	pluginName string
	event      string
	hint       string
	duration   float64
}

type inflightEventCall struct {
	event string
	unit  string
	value int
}

// newMockMetricAsyncRecorder creates a new mock metric recorder
func newMockMetricAsyncRecorder() *mockMetricAsyncRecorder {
	return &mockMetricAsyncRecorder{
		observePluginDurationCalls: make([]pluginDurationCall, 0),
		observeQueueingHintCalls:   make([]queueingHintCall, 0),
		inflightEventCalls:         make([]inflightEventCall, 0),
		IsStoppedCh:                make(chan struct{}),
	}
}

// Note: This mock provides the same methods as metrics.MetricAsyncRecorder for testing

// ObservePluginDurationAsync records plugin execution duration
func (m *mockMetricAsyncRecorder) ObservePluginDurationAsync(extensionPoint, plugin, status string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.observePluginDurationCalls = append(m.observePluginDurationCalls, pluginDurationCall{
		extensionPoint: extensionPoint,
		plugin:         plugin,
		status:         status,
		duration:       value,
	})
}

// ObserveQueueingHintDurationAsync records queueing hint duration
func (m *mockMetricAsyncRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.observeQueueingHintCalls = append(m.observeQueueingHintCalls, queueingHintCall{
		pluginName: pluginName,
		event:      event,
		hint:       hint,
		duration:   value,
	})
}

// ObserveInFlightEventsAsync records in-flight event metrics
func (m *mockMetricAsyncRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.inflightEventCalls = append(m.inflightEventCalls, inflightEventCall{
		event: eventLabel,
		unit:  "", // simplified for this mock
		value: int(valueToAdd),
	})
}

// FlushMetrics marks that flush was called
func (m *mockMetricAsyncRecorder) FlushMetrics() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.flushed = true
}

// Verification methods for tests

// getObservePluginDurationCalls returns all recorded plugin duration calls
func (m *mockMetricAsyncRecorder) getObservePluginDurationCalls() []pluginDurationCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Return a copy to avoid race conditions
	calls := make([]pluginDurationCall, len(m.observePluginDurationCalls))
	copy(calls, m.observePluginDurationCalls)
	return calls
}

// getObserveQueueingHintCalls returns all recorded queueing hint calls
func (m *mockMetricAsyncRecorder) getObserveQueueingHintCalls() []queueingHintCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	calls := make([]queueingHintCall, len(m.observeQueueingHintCalls))
	copy(calls, m.observeQueueingHintCalls)
	return calls
}

// getInflightEventCalls returns all recorded in-flight event calls
func (m *mockMetricAsyncRecorder) getInflightEventCalls() []inflightEventCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	calls := make([]inflightEventCall, len(m.inflightEventCalls))
	copy(calls, m.inflightEventCalls)
	return calls
}

// wasFlushed returns true if FlushMetrics was called
func (m *mockMetricAsyncRecorder) wasFlushed() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.flushed
}

// verifyPluginCall checks if a specific plugin duration call was made
func (m *mockMetricAsyncRecorder) verifyPluginCall(extensionPoint, plugin, status string) bool {
	calls := m.getObservePluginDurationCalls()
	for _, call := range calls {
		if call.extensionPoint == extensionPoint && call.plugin == plugin && call.status == status {
			return true
		}
	}
	return false
}
