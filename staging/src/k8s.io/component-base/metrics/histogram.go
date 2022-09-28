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

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
)

// DefBuckets is a wrapper for prometheus.DefBuckets
var DefBuckets = prometheus.DefBuckets

// LinearBuckets is a wrapper for prometheus.LinearBuckets.
func LinearBuckets(start, width float64, count int) []float64 {
	return prometheus.LinearBuckets(start, width, count)
}

// ExponentialBuckets is a wrapper for prometheus.ExponentialBuckets.
func ExponentialBuckets(start, factor float64, count int) []float64 {
	return prometheus.ExponentialBuckets(start, factor, count)
}

// Histogram is our internal representation for our wrapping struct around prometheus
// histograms. Summary implements both kubeCollector and ObserverMetric
type Histogram struct {
	ObserverMetric
	*HistogramOpts
	lazyMetric
	selfCollector
}

// NewHistogram returns an object which is Histogram-like. However, nothing
// will be measured until the histogram is registered somewhere.
func NewHistogram(opts *HistogramOpts) *Histogram {
	opts.StabilityLevel.setDefaults()

	h := &Histogram{
		HistogramOpts: opts,
		lazyMetric:    lazyMetric{},
	}
	h.setPrometheusHistogram(noopMetric{})
	h.lazyInit(h, BuildFQName(opts.Namespace, opts.Subsystem, opts.Name))
	return h
}

// setPrometheusHistogram sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (h *Histogram) setPrometheusHistogram(histogram prometheus.Histogram) {
	h.ObserverMetric = histogram
	h.initSelfCollection(histogram)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (h *Histogram) DeprecatedVersion() *semver.Version {
	return parseSemver(h.HistogramOpts.DeprecatedVersion)
}

// initializeMetric invokes the actual prometheus.Histogram object instantiation
// and stores a reference to it
func (h *Histogram) initializeMetric() {
	h.HistogramOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus gauge.
	h.setPrometheusHistogram(prometheus.NewHistogram(h.HistogramOpts.toPromHistogramOpts()))
}

// initializeDeprecatedMetric invokes the actual prometheus.Histogram object instantiation
// but modifies the Help description prior to object instantiation.
func (h *Histogram) initializeDeprecatedMetric() {
	h.HistogramOpts.markDeprecated()
	h.initializeMetric()
}

// WithContext allows the normal Histogram metric to pass in context. The context is no-op now.
func (h *Histogram) WithContext(ctx context.Context) ObserverMetric {
	return h.ObserverMetric
}

// HistogramVec is the internal representation of our wrapping struct around prometheus
// histogramVecs.
type HistogramVec struct {
	*prometheus.HistogramVec
	*HistogramOpts
	lazyMetric
	originalLabels []string
}

// NewHistogramVec returns an object which satisfies kubeCollector and wraps the
// prometheus.HistogramVec object. However, the object returned will not measure
// anything unless the collector is first registered, since the metric is lazily instantiated,
// and only members extracted after
// registration will actually measure anything.

func NewHistogramVec(opts *HistogramOpts, labels []string) *HistogramVec {
	opts.StabilityLevel.setDefaults()

	fqName := BuildFQName(opts.Namespace, opts.Subsystem, opts.Name)
	allowListLock.RLock()
	if allowList, ok := labelValueAllowLists[fqName]; ok {
		opts.LabelValueAllowLists = allowList
	}
	allowListLock.RUnlock()

	v := &HistogramVec{
		HistogramVec:   noopHistogramVec,
		HistogramOpts:  opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{},
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

// WithContext returns wrapped HistogramVec with context
func (v *HistogramVec) WithContext(ctx context.Context) *HistogramVecWithContext {
	return &HistogramVecWithContext{
		ctx:          ctx,
		HistogramVec: v,
	}
}

// HistogramVecWithContext is the wrapper of HistogramVec with context.
type HistogramVecWithContext struct {
	*HistogramVec
	ctx context.Context
}

// WithLabelValues is the wrapper of HistogramVec.WithLabelValues.
func (vc *HistogramVecWithContext) WithLabelValues(lvs ...string) ObserverMetric {
	return vc.HistogramVec.WithLabelValues(lvs...)
}

// With is the wrapper of HistogramVec.With.
func (vc *HistogramVecWithContext) With(labels map[string]string) ObserverMetric {
	return vc.HistogramVec.With(labels)
}
