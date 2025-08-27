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
	"context"
	"sync"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"go.opentelemetry.io/otel/trace"
)

// Histogram is our internal representation for our wrapping struct around prometheus histograms.
type Histogram struct {
	ObserverMetric
	*HistogramOpts
	lazyMetric
	selfCollector
}

// The metric must be register-able.
var _ Registerable = &Histogram{}

// The implementation of the Metric interface is expected by testutil.GetHistogramMetricValue.
var _ Metric = &Histogram{}

// The implementation of kubeCollector is expected for collector registration.
var _ kubeCollector = &Histogram{}

// NewHistogram returns an object which is Histogram-like. However, nothing
// will be measured until the histogram is registered somewhere.
func NewHistogram(opts *HistogramOpts) *Histogram {
	opts.StabilityLevel.setDefaults()

	h := &Histogram{
		HistogramOpts: opts,
		lazyMetric:    lazyMetric{stabilityLevel: opts.StabilityLevel},
	}
	h.setPrometheusHistogram(noopMetric{})
	h.lazyInit(h, BuildFQName(opts.Namespace, opts.Subsystem, opts.Name))
	return h
}

// initializeDeprecatedMetric invokes the actual prometheus.Histogram object instantiation
// but modifies the Help description prior to object instantiation.
func (h *Histogram) initializeDeprecatedMetric() {
	h.HistogramOpts.markDeprecated()
	h.initializeMetric()
}

// initializeMetric invokes the actual prometheus.Histogram object instantiation
// and stores a reference to it
func (h *Histogram) initializeMetric() {
	h.HistogramOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus gauge.
	h.setPrometheusHistogram(prometheus.NewHistogram(h.HistogramOpts.toPromHistogramOpts()))
}

// setPrometheusHistogram sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (h *Histogram) setPrometheusHistogram(histogram prometheus.Histogram) {
	h.ObserverMetric = histogram
	h.initSelfCollection(histogram)
}

// Desc returns the prometheus.Desc for the metric.
// This method is required by the Metric interface.
func (h *Histogram) Desc() *prometheus.Desc {
	return h.metric.Desc()
}

// Write writes the metric to the provided dto.Metric.
// This method is required by the Metric interface.
func (h *Histogram) Write(to *dto.Metric) error {
	return h.metric.Write(to)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (h *Histogram) DeprecatedVersion() *semver.Version {
	return parseSemver(h.HistogramOpts.DeprecatedVersion)
}

// Observe is the method that implements the ObserverMetric interface.
func (h *Histogram) Observe(v float64) {
	h.ObserverMetric.Observe(v)
}

// HistogramWithContext holds a context to extract exemplar labels from, and a historgram metric to attach them to. It implements the metricWithExemplar interface.
type HistogramWithContext struct {
	ctx context.Context
	*Histogram
}

// Post-equipping a histogram with context, folks should be able to use it as a regular histogram, with exemplar support.
var _ Metric = &HistogramWithContext{}
var _ metricWithExemplar = &HistogramWithContext{}

// WithContext allows a Counter to bind to a context.
func (h *Histogram) WithContext(ctx context.Context) *HistogramWithContext {
	// Return reference to a new histogram as modifying the existing one overrides the older context,
	// and blocks with semaphores. So this is a better option, see:
	// https://github.com/kubernetes/kubernetes/pull/128575#discussion_r1829509144.
	return &HistogramWithContext{
		ctx:       ctx,
		Histogram: h,
	}
}

// withExemplar attaches an exemplar to the metric.
func (e *HistogramWithContext) withExemplar(v float64) {
	if m, ok := e.Histogram.ObserverMetric.(prometheus.ExemplarObserver); ok {
		maybeSpanCtx := trace.SpanContextFromContext(e.ctx)
		if maybeSpanCtx.IsValid() && maybeSpanCtx.IsSampled() {
			exemplarLabels := prometheus.Labels{
				"trace_id": maybeSpanCtx.TraceID().String(),
				"span_id":  maybeSpanCtx.SpanID().String(),
			}
			m.ObserveWithExemplar(v, exemplarLabels)
			return
		}
	}

	e.ObserverMetric.Observe(v)
}

func (e *HistogramWithContext) Observe(v float64) {
	e.withExemplar(v)
}

// HistogramVec is the internal representation of our wrapping struct around prometheus
// histogramVecs.
type HistogramVec struct {
	*prometheus.HistogramVec
	*HistogramOpts
	lazyMetric
	originalLabels []string
}

// The implementation of kubeCollector is expected for collector registration.
// HistogramVec implements the kubeCollector interface, and not ObserverMetric interface.
var _ kubeCollector = &HistogramVec{}

// NewHistogramVec returns an object which satisfies kubeCollector and wraps the prometheus.HistogramVec object.
// However, the object returned will not measure anything unless the collector is first registered, since the metric is
// lazily instantiated, and only members extracted after registration will actually measure anything.
func NewHistogramVec(opts *HistogramOpts, labels []string) *HistogramVec {
	opts.StabilityLevel.setDefaults()

	fqName := BuildFQName(opts.Namespace, opts.Subsystem, opts.Name)

	v := &HistogramVec{
		HistogramVec:   noopHistogramVec,
		HistogramOpts:  opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{stabilityLevel: opts.StabilityLevel},
	}
	v.lazyInit(v, fqName)
	return v
}

// DeprecatedVersion returns a pointer to the Version or nil
func (v *HistogramVec) DeprecatedVersion() *semver.Version {
	return parseSemver(v.HistogramOpts.DeprecatedVersion)
}

func (v *HistogramVec) initializeMetric() {
	v.HistogramOpts.annotateStabilityLevel()
	v.HistogramVec = prometheus.NewHistogramVec(v.HistogramOpts.toPromHistogramOpts(), v.originalLabels)
}

func (v *HistogramVec) initializeDeprecatedMetric() {
	v.HistogramOpts.markDeprecated()
	v.initializeMetric()
}

// Default Prometheus Vec behavior is that member extraction results in creation of a new element
// if one with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/histogram.go#L460-L470
//
// In contrast, the Vec behavior in this package is that member extraction before registration
// returns a permanent noop object.

// WithLabelValues returns the ObserverMetric for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new ObserverMetric is created IFF the HistogramVec
// has been registered to a metrics registry.
func (v *HistogramVec) WithLabelValues(lvs ...string) ObserverMetric {
	if !v.IsCreated() {
		return noop
	}

	// Initialize label allow lists if not already initialized
	v.initializeLabelAllowListsOnce.Do(func() {
		allowListLock.RLock()
		if allowList, ok := labelValueAllowLists[v.FQName()]; ok {
			v.LabelValueAllowLists = allowList
		}
		allowListLock.RUnlock()
	})

	// Constrain label values to allowed values
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainToAllowedList(v.originalLabels, lvs)
	}
	return v.HistogramVec.WithLabelValues(lvs...)
}

// With returns the ObserverMetric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new ObserverMetric is created IFF the HistogramVec has
// been registered to a metrics registry.
func (v *HistogramVec) With(labels map[string]string) ObserverMetric {
	if !v.IsCreated() {
		return noop
	}

	// Initialize label allow lists if not already initialized
	v.initializeLabelAllowListsOnce.Do(func() {
		allowListLock.RLock()
		if allowList, ok := labelValueAllowLists[v.FQName()]; ok {
			v.LabelValueAllowLists = allowList
		}
		allowListLock.RUnlock()
	})

	// Constrain label map to allowed values
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainLabelMap(labels)
	}

	return v.HistogramVec.With(labels)
}

// Delete deletes the metric where the variable labels are the same as those
// passed in as labels. It returns true if a metric was deleted.
//
// It is not an error if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc. However, such inconsistent Labels
// can never match an actual metric, so the method will always return false in
// that case.
func (v *HistogramVec) Delete(labels map[string]string) bool {
	if !v.IsCreated() {
		return false // since we haven't created the metric, we haven't deleted a metric with the passed in values
	}
	return v.HistogramVec.Delete(labels)
}

// Reset deletes all metrics in this vector.
func (v *HistogramVec) Reset() {
	if !v.IsCreated() {
		return
	}

	v.HistogramVec.Reset()
}

// ResetLabelAllowLists resets the label allow list for the HistogramVec.
// NOTE: This should only be used in test.
func (v *HistogramVec) ResetLabelAllowLists() {
	v.initializeLabelAllowListsOnce = sync.Once{}
	v.LabelValueAllowLists = nil
}

// WithContext is a no-op for now, users should still attach this to vectors for seamless future API upgrades (which rely on the context).
// This is done to keep extensions (such as exemplars) on the counter and not its derivatives.
// Note that there are no actual uses for this in the codebase except for chaining it with `WithLabelValues`, which makes no use of the context.
// Furthermore, Prometheus, which is upstream from this library, does not support contextual vectors, so we don't want to diverge.
func (v *HistogramVec) WithContext(_ context.Context) *HistogramVec {
	return v
}
