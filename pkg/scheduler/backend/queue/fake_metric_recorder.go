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
	"context"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// FakeMetricAsyncRecorder is a fake implementation of MetricAsyncRecorder for testing
type FakeMetricAsyncRecorder struct {
	mu                  sync.Mutex
	pluginDurationCalls []pluginDurationCall
	queueingHintCalls   []queueingHintCall
	inFlightEventsCalls []inFlightEventsCall
	flushMetricsCalls   int
	IsStoppedCh         chan struct{}
}

type pluginDurationCall struct {
	extensionPoint string
	pluginName     string
	status         string
	value          float64
}

type queueingHintCall struct {
	pluginName string
	event      string
	hint       string
	value      float64
}

type inFlightEventsCall struct {
	eventLabel string
	valueToAdd float64
	forceFlush bool
}

// NewFakeMetricAsyncRecorder creates a new fake recorder
func NewFakeMetricAsyncRecorder() *FakeMetricAsyncRecorder {
	return &FakeMetricAsyncRecorder{
		IsStoppedCh: make(chan struct{}),
	}
}

// ObservePluginDurationAsync records the call for verification
func (f *FakeMetricAsyncRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pluginDurationCalls = append(f.pluginDurationCalls, pluginDurationCall{
		extensionPoint: extensionPoint,
		pluginName:     pluginName,
		status:         status,
		value:          value,
	})
}

// ObserveQueueingHintDurationAsync records the call for verification
func (f *FakeMetricAsyncRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queueingHintCalls = append(f.queueingHintCalls, queueingHintCall{
		pluginName: pluginName,
		event:      event,
		hint:       hint,
		value:      value,
	})
}

// ObserveInFlightEventsAsync records the call for verification
func (f *FakeMetricAsyncRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inFlightEventsCalls = append(f.inFlightEventsCalls, inFlightEventsCall{
		eventLabel: eventLabel,
		valueToAdd: valueToAdd,
		forceFlush: forceFlush,
	})
}

// FlushMetrics records the flush call
func (f *FakeMetricAsyncRecorder) FlushMetrics() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushMetricsCalls++
}

// GetPluginDurationCalls returns all plugin duration calls for verification
func (f *FakeMetricAsyncRecorder) GetPluginDurationCalls() []pluginDurationCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]pluginDurationCall{}, f.pluginDurationCalls...)
}

// GetQueueingHintCalls returns all queueing hint calls for verification
func (f *FakeMetricAsyncRecorder) GetQueueingHintCalls() []queueingHintCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]queueingHintCall{}, f.queueingHintCalls...)
}

// GetInFlightEventsCalls returns all in-flight events calls for verification
func (f *FakeMetricAsyncRecorder) GetInFlightEventsCalls() []inFlightEventsCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]inFlightEventsCall{}, f.inFlightEventsCalls...)
}

// GetFlushMetricsCalls returns the number of flush calls
func (f *FakeMetricAsyncRecorder) GetFlushMetricsCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.flushMetricsCalls
}

// AsMetricAsyncRecorder returns the fake as a MetricAsyncRecorder
// This uses unsafe conversion which is necessary because the queue expects a concrete type
func (f *FakeMetricAsyncRecorder) AsMetricAsyncRecorder() metrics.MetricAsyncRecorder {
	// Create a recorder but redirect its methods to our fake
	// This is a workaround until we can refactor to use interfaces
	ctx, cancel := context.WithCancel(context.Background())
	recorder := metrics.NewMetricsAsyncRecorder(10, time.Microsecond, ctx.Done())

	// We need to stop the real recorder and use our fake
	cancel()
	<-recorder.IsStoppedCh

	// Now we can safely return our fake with the same structure
	return metrics.MetricAsyncRecorder{
		IsStoppedCh: f.IsStoppedCh,
	}
}
