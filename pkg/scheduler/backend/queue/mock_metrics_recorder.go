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
	"github.com/stretchr/testify/mock"
)

// MockMetricsRecorder is a mock implementation of metrics.MetricAsyncRecorder using testify/mock
type MockMetricsRecorder struct {
	mock.Mock
}

// NewMockMetricsRecorder creates a new mock recorder
func NewMockMetricsRecorder() *MockMetricsRecorder {
	return &MockMetricsRecorder{}
}

// ObservePluginDurationAsync mocks the metric observation
func (m *MockMetricsRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	m.Called(extensionPoint, pluginName, status, value)
}

// ObserveQueueingHintDurationAsync mocks the queueing hint metric observation
func (m *MockMetricsRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	m.Called(pluginName, event, hint, value)
}

// ObserveInFlightEventsAsync mocks the in-flight events metric observation
func (m *MockMetricsRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	m.Called(eventLabel, valueToAdd, forceFlush)
}

// FlushMetrics mocks the flush operation
func (m *MockMetricsRecorder) FlushMetrics() {
	m.Called()
}
