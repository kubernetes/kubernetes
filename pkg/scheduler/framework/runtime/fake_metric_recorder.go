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
	"time"
	"unsafe"

	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// PluginDurationCall records a call to ObservePluginDurationAsync
type PluginDurationCall struct {
	ExtensionPoint string
	PluginName     string
	Status         string
	Duration       float64
}

// QueueingHintCall records a call to ObserveQueueingHintDurationAsync
type QueueingHintCall struct {
	PluginName string
	Event      string
	Hint       string
	Duration   float64
}

// InflightEventCall records a call to ObserveInFlightEventsAsync
type InflightEventCall struct {
	EventLabel string
	Value      float64
	ForceFlush bool
}

// FakeMetricAsyncRecorder is a fake implementation that records calls instead of metrics
// It has the same memory layout as metrics.MetricAsyncRecorder to enable unsafe casting
type FakeMetricAsyncRecorder struct {
	// These fields match the exact layout of metrics.MetricAsyncRecorder
	// This is required for the unsafe casting to work properly

	// From metrics.MetricAsyncRecorder:
	bufferCh                                   chan any // Using any to match any channel type
	bufferSize                                 int
	interval                                   time.Duration
	aggregatedInflightEventMetric              map[any]int // Using any to match gaugeVecMetricKey
	aggregatedInflightEventMetricLastFlushTime time.Time
	aggregatedInflightEventMetricBufferCh      chan any // Using any to match *gaugeVecMetric
	stopCh                                     <-chan struct{}
	IsStoppedCh                                chan struct{}

	// Our additional fields for capturing calls
	mu                  sync.Mutex
	pluginDurationCalls []PluginDurationCall
	queueingHintCalls   []QueueingHintCall
	inflightEventCalls  []InflightEventCall
	flushCount          int
}

// NewFakeMetricAsyncRecorder creates a new fake recorder for testing
func NewFakeMetricAsyncRecorder() *FakeMetricAsyncRecorder {
	stopCh := make(chan struct{})
	close(stopCh) // Close immediately to prevent goroutine start

	return &FakeMetricAsyncRecorder{
		bufferCh:                      make(chan any, 1),
		bufferSize:                    1,
		interval:                      time.Millisecond,
		aggregatedInflightEventMetric: make(map[any]int),
		aggregatedInflightEventMetricLastFlushTime: time.Now(),
		aggregatedInflightEventMetricBufferCh:      make(chan any, 1),
		stopCh:                                     stopCh,
		IsStoppedCh:                                make(chan struct{}),
		pluginDurationCalls:                        make([]PluginDurationCall, 0),
		queueingHintCalls:                          make([]QueueingHintCall, 0),
		inflightEventCalls:                         make([]InflightEventCall, 0),
	}
}

// ObservePluginDurationAsync records the call instead of recording metrics
func (f *FakeMetricAsyncRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls = append(f.pluginDurationCalls, PluginDurationCall{
		ExtensionPoint: extensionPoint,
		PluginName:     pluginName,
		Status:         status,
		Duration:       value,
	})
}

// ObserveQueueingHintDurationAsync records the call instead of recording metrics
func (f *FakeMetricAsyncRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.queueingHintCalls = append(f.queueingHintCalls, QueueingHintCall{
		PluginName: pluginName,
		Event:      event,
		Hint:       hint,
		Duration:   value,
	})
}

// ObserveInFlightEventsAsync records the call instead of recording metrics
func (f *FakeMetricAsyncRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.inflightEventCalls = append(f.inflightEventCalls, InflightEventCall{
		EventLabel: eventLabel,
		Value:      valueToAdd,
		ForceFlush: forceFlush,
	})
}

// FlushMetrics records that flush was called
func (f *FakeMetricAsyncRecorder) FlushMetrics() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushCount++
}

// GetPluginDurationCalls returns all recorded plugin duration calls
func (f *FakeMetricAsyncRecorder) GetPluginDurationCalls() []PluginDurationCall {
	f.mu.Lock()
	defer f.mu.Unlock()

	result := make([]PluginDurationCall, len(f.pluginDurationCalls))
	copy(result, f.pluginDurationCalls)
	return result
}

// GetQueueingHintCalls returns all recorded queueing hint calls
func (f *FakeMetricAsyncRecorder) GetQueueingHintCalls() []QueueingHintCall {
	f.mu.Lock()
	defer f.mu.Unlock()

	result := make([]QueueingHintCall, len(f.queueingHintCalls))
	copy(result, f.queueingHintCalls)
	return result
}

// GetInflightEventCalls returns all recorded inflight event calls
func (f *FakeMetricAsyncRecorder) GetInflightEventCalls() []InflightEventCall {
	f.mu.Lock()
	defer f.mu.Unlock()

	result := make([]InflightEventCall, len(f.inflightEventCalls))
	copy(result, f.inflightEventCalls)
	return result
}

// GetFlushCount returns how many times FlushMetrics was called
func (f *FakeMetricAsyncRecorder) GetFlushCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.flushCount
}

// VerifyPluginCall checks if a specific plugin call was made
func (f *FakeMetricAsyncRecorder) VerifyPluginCall(extensionPoint, pluginName, status string) bool {
	calls := f.GetPluginDurationCalls()
	for _, call := range calls {
		if call.ExtensionPoint == extensionPoint &&
			call.PluginName == pluginName &&
			call.Status == status {
			return true
		}
	}
	return false
}

// Reset clears all recorded calls (useful for test cleanup)
func (f *FakeMetricAsyncRecorder) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.pluginDurationCalls = f.pluginDurationCalls[:0]
	f.queueingHintCalls = f.queueingHintCalls[:0]
	f.inflightEventCalls = f.inflightEventCalls[:0]
	f.flushCount = 0
}

// AsMetricAsyncRecorder returns this fake recorder cast to *metrics.MetricAsyncRecorder
// This works because the memory layout matches
func (f *FakeMetricAsyncRecorder) AsMetricAsyncRecorder() *metrics.MetricAsyncRecorder {
	return (*metrics.MetricAsyncRecorder)(unsafe.Pointer(f))
}
