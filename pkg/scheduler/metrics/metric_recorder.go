/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"time"

	"k8s.io/component-base/metrics"
	fwk "k8s.io/kube-scheduler/framework"
)

// Entity is either a pod or a podgroup that gets recorded in the metrics.
type Entity interface {
	// Size returns the number of pods within the entity (1 for a single pod, or the pod count for a pod group).
	Size() int
	// Type returns the entity type (e.g. "pod" or "podgroup").
	Type() fwk.EntityKeyType
}

// MetricRecorder represents a metric recorder which maintains a metric through Add(), Remove(), Update(), and Clear().
type MetricRecorder interface {
	// Add records the addition of an entity.
	Add(entity Entity)
	// Remove records the removal of an entity.
	Remove(entity Entity)
	// Update records the update of an entity.
	Update(oldEntity, newEntity Entity)
	// Clear resets recorded metrics to zero.
	Clear()
}

var _ MetricRecorder = &QueuedEntitiesRecorder{}

// QueuedEntitiesRecorder is an implementation of MetricRecorder.
type QueuedEntitiesRecorder struct {
	pods     metrics.GaugeMetric
	entities func(entityType string) metrics.GaugeMetric
}

// NewActiveEntitiesRecorder returns ActivePods and ActiveEntities in a Prometheus metric fashion.
func NewActiveEntitiesRecorder() *QueuedEntitiesRecorder {
	return &QueuedEntitiesRecorder{
		pods:     ActivePods(),
		entities: ActiveEntities,
	}
}

// NewUnschedulableEntitiesRecorder returns UnschedulablePods and UnschedulableEntities in a Prometheus metric fashion.
func NewUnschedulableEntitiesRecorder() *QueuedEntitiesRecorder {
	return &QueuedEntitiesRecorder{
		pods:     UnschedulablePods(),
		entities: UnschedulableEntities,
	}
}

// NewBackoffEntitiesRecorder returns BackoffPods and BackoffEntities in a Prometheus metric fashion.
func NewBackoffEntitiesRecorder() *QueuedEntitiesRecorder {
	return &QueuedEntitiesRecorder{
		pods:     BackoffPods(),
		entities: BackoffEntities,
	}
}

// NewGatedEntitiesRecorder returns GatedPods and GatedEntities in a Prometheus metric fashion.
func NewGatedEntitiesRecorder() *QueuedEntitiesRecorder {
	return &QueuedEntitiesRecorder{
		pods:     GatedPods(),
		entities: GatedEntities,
	}
}

// EntityToLabel converts an Entity to an entity metric label string.
func EntityToLabel(entity Entity) (string, bool) {
	if entity == nil {
		return "", false
	}

	switch entity.Type() {
	case fwk.PodKeyType:
		return Pod, true
	case fwk.PodGroupKeyType:
		return PodGroup, true
	}

	return "", false
}

// Add records the addition of an entity.
// It increments pending pods and queued entities metric counters.
func (r *QueuedEntitiesRecorder) Add(entity Entity) {
	r.pods.Add(float64(entity.Size()))
	if label, ok := EntityToLabel(entity); ok {
		r.entities(label).Inc()
	}
}

// Remove records the removal of an entity.
// It decrements pending pods and queued entities metric counters.
func (r *QueuedEntitiesRecorder) Remove(entity Entity) {
	r.pods.Add(-float64(entity.Size()))
	if label, ok := EntityToLabel(entity); ok {
		r.entities(label).Dec()
	}
}

// Update records the update of an entity.
// It updates pending pods metric counter only.
// It shouldn't update queued entities metric, because the entity type cannot be changed, and an entity will always be a single object.
func (r *QueuedEntitiesRecorder) Update(oldEntity, newEntity Entity) {
	diff := newEntity.Size() - oldEntity.Size()
	r.pods.Add(float64(diff))
}

// Clear resets pending pods and queued entities metric counters to 0.
func (r *QueuedEntitiesRecorder) Clear() {
	r.pods.Set(float64(0))
	r.entities(Pod).Set(float64(0))
	r.entities(PodGroup).Set(float64(0))
}

// histogramVecMetric is the data structure passed in the buffer channel between the main framework thread
// and the metricsRecorder goroutine.
type histogramVecMetric struct {
	metric      *metrics.HistogramVec
	labelValues []string
	value       float64
}

type gaugeVecMetric struct {
	metric      *metrics.GaugeVec
	labelValues []string
	valueToAdd  float64
}

type gaugeVecMetricKey struct {
	metricName string
	labelValue string
}

// MetricAsyncRecorder records metric in a separate goroutine to avoid overhead in the critical path.
type MetricAsyncRecorder struct {
	// bufferCh is a channel that serves as a metrics buffer before the metricsRecorder goroutine reports it.
	bufferCh chan *histogramVecMetric
	// if bufferSize is reached, incoming metrics will be discarded.
	bufferSize int
	// how often the recorder runs to flush the metrics.
	interval time.Duration

	// aggregatedInflightEventMetric is only to record InFlightEvents metric asynchronously.
	// It's a map from gaugeVecMetricKey to the aggregated value
	// and the aggregated value is flushed to Prometheus every time the interval is reached.
	// Note that we don't lock the map deliberately because we assume the queue takes lock before updating the in-flight events.
	aggregatedInflightEventMetric              map[gaugeVecMetricKey]int
	aggregatedInflightEventMetricLastFlushTime time.Time
	aggregatedInflightEventMetricBufferCh      chan *gaugeVecMetric

	// stopCh is used to stop the goroutine which periodically flushes metrics.
	stopCh <-chan struct{}
	// IsStoppedCh indicates whether the goroutine is stopped. It's used in tests only to make sure
	// the metric flushing goroutine is stopped so that tests can collect metrics for verification.
	IsStoppedCh chan struct{}
}

func NewMetricsAsyncRecorder(bufferSize int, interval time.Duration, stopCh <-chan struct{}) *MetricAsyncRecorder {
	recorder := &MetricAsyncRecorder{
		bufferCh:                      make(chan *histogramVecMetric, bufferSize),
		bufferSize:                    bufferSize,
		interval:                      interval,
		stopCh:                        stopCh,
		aggregatedInflightEventMetric: make(map[gaugeVecMetricKey]int),
		aggregatedInflightEventMetricLastFlushTime: time.Now(),
		aggregatedInflightEventMetricBufferCh:      make(chan *gaugeVecMetric, bufferSize),
		IsStoppedCh:                                make(chan struct{}),
	}
	go recorder.run()
	return recorder
}

// StoppedCh returns a channel that is closed when the recorder's background goroutine has stopped.
func (r *MetricAsyncRecorder) StoppedCh() <-chan struct{} {
	return r.IsStoppedCh
}

// ObserveFrameworkExtensionPointDurationAsync observes the framework_extension_point_duration_seconds metric.
// The metric will be flushed to Prometheus asynchronously.
func (r *MetricAsyncRecorder) ObserveFrameworkExtensionPointDurationAsync(extensionPoint, status, profileName string, value float64) {
	r.observeMetricAsync(FrameworkExtensionPointDuration, value, extensionPoint, status, profileName)
}

// ObservePluginDurationAsync observes the plugin_execution_duration_seconds metric.
// The metric will be flushed to Prometheus asynchronously.
func (r *MetricAsyncRecorder) ObservePluginDurationAsync(extensionPoint, pluginName, status string, value float64) {
	r.observeMetricAsync(PluginExecutionDuration, value, pluginName, extensionPoint, status)
}

// ObserveQueueingHintDurationAsync observes the queueing_hint_execution_duration_seconds metric.
// The metric will be flushed to Prometheus asynchronously.
func (r *MetricAsyncRecorder) ObserveQueueingHintDurationAsync(pluginName, event, hint string, value float64) {
	r.observeMetricAsync(queueingHintExecutionDuration, value, pluginName, event, hint)
}

// ObserveInFlightEventsAsync observes the in_flight_events metric.
//
// Note that this function is not goroutine-safe;
// we don't lock the map deliberately for the performance reason and we assume the queue (i.e., the caller) takes lock before updating the in-flight events.
func (r *MetricAsyncRecorder) ObserveInFlightEventsAsync(eventLabel string, valueToAdd float64, forceFlush bool) {
	r.aggregatedInflightEventMetric[gaugeVecMetricKey{metricName: InFlightEvents.Name, labelValue: eventLabel}] += int(valueToAdd)

	// Only flush the metric to the channel if the interval is reached.
	// The values are flushed to Prometheus in the run() function, which runs once the interval time.
	// Note: we implement this flushing here, not in FlushMetrics, because, if we did so, we would need to implement a lock for the map, which we want to avoid.
	if forceFlush || time.Since(r.aggregatedInflightEventMetricLastFlushTime) > r.interval {
		for key, value := range r.aggregatedInflightEventMetric {
			newMetric := &gaugeVecMetric{
				metric:      InFlightEvents,
				labelValues: []string{key.labelValue},
				valueToAdd:  float64(value),
			}
			select {
			case r.aggregatedInflightEventMetricBufferCh <- newMetric:
			default:
			}
		}
		r.aggregatedInflightEventMetricLastFlushTime = time.Now()
		// reset
		r.aggregatedInflightEventMetric = make(map[gaugeVecMetricKey]int)
	}
}

func (r *MetricAsyncRecorder) observeMetricAsync(m *metrics.HistogramVec, value float64, labelsValues ...string) {
	newMetric := &histogramVecMetric{
		metric:      m,
		labelValues: labelsValues,
		value:       value,
	}
	select {
	case r.bufferCh <- newMetric:
	default:
	}
}

// run flushes buffered metrics into Prometheus every second.
func (r *MetricAsyncRecorder) run() {
	for {
		r.FlushMetrics()
		select {
		case <-r.stopCh:
			close(r.IsStoppedCh)
			return
		case <-time.After(r.interval):
		}
	}
}

// FlushMetrics tries to clean up the bufferCh by reading at most bufferSize metrics.
func (r *MetricAsyncRecorder) FlushMetrics() {
	for i := 0; i < r.bufferSize; i++ {
		select {
		case m := <-r.bufferCh:
			m.metric.WithLabelValues(m.labelValues...).Observe(m.value)
		default:
			// no more value
		}

		select {
		case m := <-r.aggregatedInflightEventMetricBufferCh:
			m.metric.WithLabelValues(m.labelValues...).Add(m.valueToAdd)
		default:
			// no more value
		}
	}
}
