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

package queue

import (
	"sync"
)

// MockMetricAsyncRecorder is a mock implementation of MetricAsyncRecorder for testing
type MockMetricAsyncRecorder struct {
	mu                  sync.Mutex
	pluginDurationCalls []pluginDurationCall
	IsStoppedCh         chan struct{}
}

type pluginDurationCall struct {
	extensionPoint string
	pluginName     string
	status         string
	value          float64
}

// NewMockMetricAsyncRecorder creates a new mock recorder
func NewMockMetricAsyncRecorder() *MockMetricAsyncRecorder {
	return &MockMetricAsyncRecorder{
		IsStoppedCh: make(chan struct{}),
	}
}

// ObservePluginDurationAsync records the call for verification
func (m *MockMetricAsyncRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pluginDurationCalls = append(m.pluginDurationCalls, pluginDurationCall{
		extensionPoint: extensionPoint,
		pluginName:     pluginName,
		status:         status,
		value:          value,
	})
}

// ObserveQueueingHintDurationAsync is a noop stub (not verified in tests)
func (m *MockMetricAsyncRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
}

// ObserveInFlightEventsAsync is a noop stub (not verified in tests)
func (m *MockMetricAsyncRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
}

// FlushMetrics is a noop stub (not verified in tests)
func (m *MockMetricAsyncRecorder) FlushMetrics() {
}

// GetPluginDurationCalls returns all plugin duration calls for verification
func (m *MockMetricAsyncRecorder) GetPluginDurationCalls() []pluginDurationCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]pluginDurationCall{}, m.pluginDurationCalls...)
}
