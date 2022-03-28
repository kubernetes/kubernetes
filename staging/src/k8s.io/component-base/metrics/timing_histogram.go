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

	"github.com/blang/semver"
	thprom "k8s.io/component-base/metrics/prometheusextension"
)

// TimingHistogram is our internal representation for our wrapping struct around prometheus
// TimingHistograms. It implements both kubeCollector and WritableVariable
type TimingHistogram struct {
	WritableVariable
	*TimingHistogramOpts
	lazyMetric
	selfCollector
}

// NewTimingHistogram returns an object which is TimingHistogram-like. However, nothing
// will be measured until the histogram is registered somewhere.
func NewTimingHistogram(opts *TimingHistogramOpts) *TimingHistogram {
	opts.StabilityLevel.setDefaults()

	h := &TimingHistogram{
		TimingHistogramOpts: opts,
		lazyMetric:          lazyMetric{},
	}
	h.setPrometheusHistogram(noopMetric{})
	h.lazyInit(h, BuildFQName(opts.Namespace, opts.Subsystem, opts.Name))
	return h
}

// setPrometheusHistogram sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (h *TimingHistogram) setPrometheusHistogram(histogram thprom.TimingHistogram) {
	h.WritableVariable = histogram
	h.initSelfCollection(histogram)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (h *TimingHistogram) DeprecatedVersion() *semver.Version {
	return parseSemver(h.TimingHistogramOpts.DeprecatedVersion)
}

// initializeMetric invokes the actual prometheus.Histogram object instantiation
// and stores a reference to it
func (h *TimingHistogram) initializeMetric() {
	h.TimingHistogramOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus metric.
	under, err := thprom.NewTimingHistogram(h.TimingHistogramOpts.toPromTimingHistogramOpts())
	if err != nil {
		panic(err)
	}
	h.setPrometheusHistogram(under)
}

// initializeDeprecatedMetric invokes the actual prometheus.Histogram object instantiation
// but modifies the Help description prior to object instantiation.
func (h *TimingHistogram) initializeDeprecatedMetric() {
	h.TimingHistogramOpts.markDeprecated()
	h.initializeMetric()
}

// WithContext allows the normal Histogram metric to pass in context. The context is no-op now.
func (h *TimingHistogram) WithContext(ctx context.Context) WritableVariable {
	return h.WritableVariable
}

// TimingHistogramVec is the internal representation of our wrapping struct around prometheus
// TimingHistogramVecs.
type TimingHistogramVec struct {
	*thprom.TimingHistogramVec
	*TimingHistogramOpts
	lazyMetric
	originalLabels []string
}

// NewTimingHistogramVec returns an object which satisfies kubeCollector and wraps the
// thprom.TimingHistogramVec object. However, the object returned will not measure
// anything unless the collector is first registered, since the metric is lazily instantiated.
func NewTimingHistogramVec(opts *TimingHistogramOpts, labels []string) *TimingHistogramVec {
	opts.StabilityLevel.setDefaults()

	fqName := BuildFQName(opts.Namespace, opts.Subsystem, opts.Name)
	allowListLock.RLock()
	if allowList, ok := labelValueAllowLists[fqName]; ok {
		opts.LabelValueAllowLists = allowList
	}
	allowListLock.RUnlock()

	v := &TimingHistogramVec{
		TimingHistogramVec:  noopTimingHistogramVec,
		TimingHistogramOpts: opts,
		originalLabels:      labels,
		lazyMetric:          lazyMetric{},
	}
	v.lazyInit(v, fqName)
	return v
}

// DeprecatedVersion returns a pointer to the Version or nil
func (v *TimingHistogramVec) DeprecatedVersion() *semver.Version {
	return parseSemver(v.TimingHistogramOpts.DeprecatedVersion)
}

func (v *TimingHistogramVec) initializeMetric() {
	v.TimingHistogramOpts.annotateStabilityLevel()
	v.TimingHistogramVec = thprom.NewTimingHistogramVec(v.TimingHistogramOpts.toPromTimingHistogramOpts(), v.originalLabels)
}

func (v *TimingHistogramVec) initializeDeprecatedMetric() {
	v.TimingHistogramOpts.markDeprecated()
	v.initializeMetric()
}

// Default Prometheus behavior actually results in the creation of a new metric
// if a metric with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/histogram.go#L460-L470

// WithLabelValues returns the WritableVariable for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new WritableVariable is created IFF the HistogramVec
// has been registered to a metrics registry.
func (v *TimingHistogramVec) WithLabelValues(lvs ...string) WritableVariable {
	if !v.IsCreated() {
		return noop
	}
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainToAllowedList(v.originalLabels, lvs)
	}
	return v.TimingHistogramVec.WithLabelValues(lvs...)
}

// With returns the ObserverMetric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new ObserverMetric is created IFF the HistogramVec has
// been registered to a metrics registry.
func (v *TimingHistogramVec) With(labels map[string]string) WritableVariable {
	if !v.IsCreated() {
		return noop
	}
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainLabelMap(labels)
	}
	return v.TimingHistogramVec.With(labels)
}

// Delete deletes the metric where the variable labels are the same as those
// passed in as labels. It returns true if a metric was deleted.
//
// It is not an error if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc. However, such inconsistent Labels
// can never match an actual metric, so the method will always return false in
// that case.
func (v *TimingHistogramVec) Delete(labels map[string]string) bool {
	if !v.IsCreated() {
		return false // since we haven't created the metric, we haven't deleted a metric with the passed in values
	}
	return v.TimingHistogramVec.Delete(labels)
}

// Reset deletes all metrics in this vector.
func (v *TimingHistogramVec) Reset() {
	if !v.IsCreated() {
		return
	}

	v.TimingHistogramVec.Reset()
}

// WithContext returns wrapped HistogramVec with context
func (v *TimingHistogramVec) WithContext(ctx context.Context) *TimingHistogramVecWithContext {
	return &TimingHistogramVecWithContext{
		ctx:                ctx,
		TimingHistogramVec: *v,
	}
}

// HistogramVecWithContext is the wrapper of HistogramVec with context.
type TimingHistogramVecWithContext struct {
	TimingHistogramVec
	ctx context.Context
}

// WithLabelValues is the wrapper of HistogramVec.WithLabelValues.
func (vc *TimingHistogramVecWithContext) WithLabelValues(lvs ...string) WritableVariable {
	return vc.TimingHistogramVec.WithLabelValues(lvs...)
}

// With is the wrapper of HistogramVec.With.
func (vc *TimingHistogramVecWithContext) With(labels map[string]string) WritableVariable {
	return vc.TimingHistogramVec.With(labels)
}
