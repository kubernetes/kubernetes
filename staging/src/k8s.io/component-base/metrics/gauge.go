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

	"k8s.io/component-base/version"
)

// Gauge is our internal representation for our wrapping struct around prometheus
// gauges. kubeGauge implements both kubeCollector and KubeGauge.
type Gauge struct {
	GaugeMetric
	*GaugeOpts
	lazyMetric
	selfCollector
}

var _ GaugeMetric = &Gauge{}
var _ Registerable = &Gauge{}
var _ kubeCollector = &Gauge{}

// NewGauge returns an object which satisfies the kubeCollector, Registerable, and Gauge interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated.
func NewGauge(opts *GaugeOpts) *Gauge {
	opts.StabilityLevel.setDefaults()

	kc := &Gauge{
		GaugeOpts:  opts,
		lazyMetric: lazyMetric{stabilityLevel: opts.StabilityLevel},
	}
	kc.setPrometheusGauge(noop)
	kc.lazyInit(kc, BuildFQName(opts.Namespace, opts.Subsystem, opts.Name))
	return kc
}

// setPrometheusGauge sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (g *Gauge) setPrometheusGauge(gauge prometheus.Gauge) {
	g.GaugeMetric = gauge
	g.initSelfCollection(gauge)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (g *Gauge) DeprecatedVersion() *semver.Version {
	return parseSemver(g.GaugeOpts.DeprecatedVersion)
}

// initializeMetric invocation creates the actual underlying Gauge. Until this method is called
// the underlying gauge is a no-op.
func (g *Gauge) initializeMetric() {
	g.GaugeOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus gauge.
	g.setPrometheusGauge(prometheus.NewGauge(g.GaugeOpts.toPromGaugeOpts()))
}

// initializeDeprecatedMetric invocation creates the actual (but deprecated) Gauge. Until this method
// is called the underlying gauge is a no-op.
func (g *Gauge) initializeDeprecatedMetric() {
	g.GaugeOpts.markDeprecated()
	g.initializeMetric()
}

// WithContext allows the normal Gauge metric to pass in context. The context is no-op now.
func (g *Gauge) WithContext(ctx context.Context) GaugeMetric {
	return g.GaugeMetric
}

// GaugeVec is the internal representation of our wrapping struct around prometheus
// gaugeVecs. kubeGaugeVec implements both kubeCollector and KubeGaugeVec.
type GaugeVec struct {
	*prometheus.GaugeVec
	*GaugeOpts
	lazyMetric
	originalLabels []string
}

var _ GaugeVecMetric = &GaugeVec{}
var _ Registerable = &GaugeVec{}
var _ kubeCollector = &GaugeVec{}

// NewGaugeVec returns an object which satisfies the kubeCollector, Registerable, and GaugeVecMetric interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated, and only members extracted after
// registration will actually measure anything.
func NewGaugeVec(opts *GaugeOpts, labels []string) *GaugeVec {
	opts.StabilityLevel.setDefaults()

	fqName := BuildFQName(opts.Namespace, opts.Subsystem, opts.Name)

	cv := &GaugeVec{
		GaugeVec:       noopGaugeVec,
		GaugeOpts:      opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{stabilityLevel: opts.StabilityLevel},
	}
	cv.lazyInit(cv, fqName)
	return cv
}

// DeprecatedVersion returns a pointer to the Version or nil
func (v *GaugeVec) DeprecatedVersion() *semver.Version {
	return parseSemver(v.GaugeOpts.DeprecatedVersion)
}

// initializeMetric invocation creates the actual underlying GaugeVec. Until this method is called
// the underlying gaugeVec is a no-op.
func (v *GaugeVec) initializeMetric() {
	v.GaugeOpts.annotateStabilityLevel()
	v.GaugeVec = prometheus.NewGaugeVec(v.GaugeOpts.toPromGaugeOpts(), v.originalLabels)
}

// initializeDeprecatedMetric invocation creates the actual (but deprecated) GaugeVec. Until this method is called
// the underlying gaugeVec is a no-op.
func (v *GaugeVec) initializeDeprecatedMetric() {
	v.GaugeOpts.markDeprecated()
	v.initializeMetric()
}

func (v *GaugeVec) WithLabelValuesChecked(lvs ...string) (GaugeMetric, error) {
	if !v.IsCreated() {
		if v.IsHidden() {
			return noop, nil
		}
		return noop, errNotRegistered // return no-op gauge
	}
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainToAllowedList(v.originalLabels, lvs)
	} else {
		v.initializeLabelAllowListsOnce.Do(func() {
			allowListLock.RLock()
			if allowList, ok := labelValueAllowLists[v.FQName()]; ok {
				v.LabelValueAllowLists = allowList
				allowList.ConstrainToAllowedList(v.originalLabels, lvs)
			}
			allowListLock.RUnlock()
		})
	}
	elt, err := v.GaugeVec.GetMetricWithLabelValues(lvs...)
	return elt, err
}

// Default Prometheus Vec behavior is that member extraction results in creation of a new element
// if one with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/gauge.go#L190-L208
//
// In contrast, the Vec behavior in this package is that member extraction before registration
// returns a permanent noop object.

// WithLabelValues returns the GaugeMetric for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new GaugeMetric is created IFF the gaugeVec
// has been registered to a metrics registry.
func (v *GaugeVec) WithLabelValues(lvs ...string) GaugeMetric {
	ans, err := v.WithLabelValuesChecked(lvs...)
	if err == nil || ErrIsNotRegistered(err) {
		return ans
	}
	panic(err)
}

func (v *GaugeVec) WithChecked(labels map[string]string) (GaugeMetric, error) {
	if !v.IsCreated() {
		if v.IsHidden() {
			return noop, nil
		}
		return noop, errNotRegistered // return no-op gauge
	}
	if v.LabelValueAllowLists != nil {
		v.LabelValueAllowLists.ConstrainLabelMap(labels)
	} else {
		v.initializeLabelAllowListsOnce.Do(func() {
			allowListLock.RLock()
			if allowList, ok := labelValueAllowLists[v.FQName()]; ok {
				v.LabelValueAllowLists = allowList
				allowList.ConstrainLabelMap(labels)
			}
			allowListLock.RUnlock()
		})
	}
	elt, err := v.GaugeVec.GetMetricWith(labels)
	return elt, err
}

// With returns the GaugeMetric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new GaugeMetric is created IFF the gaugeVec has
// been registered to a metrics registry.
func (v *GaugeVec) With(labels map[string]string) GaugeMetric {
	ans, err := v.WithChecked(labels)
	if err == nil || ErrIsNotRegistered(err) {
		return ans
	}
	panic(err)
}

// Delete deletes the metric where the variable labels are the same as those
// passed in as labels. It returns true if a metric was deleted.
//
// It is not an error if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc. However, such inconsistent Labels
// can never match an actual metric, so the method will always return false in
// that case.
func (v *GaugeVec) Delete(labels map[string]string) bool {
	if !v.IsCreated() {
		return false // since we haven't created the metric, we haven't deleted a metric with the passed in values
	}
	return v.GaugeVec.Delete(labels)
}

// Reset deletes all metrics in this vector.
func (v *GaugeVec) Reset() {
	if !v.IsCreated() {
		return
	}

	v.GaugeVec.Reset()
}

// ResetLabelAllowLists resets the label allow list for the GaugeVec.
// NOTE: This should only be used in test.
func (v *GaugeVec) ResetLabelAllowLists() {
	v.initializeLabelAllowListsOnce = sync.Once{}
	v.LabelValueAllowLists = nil
}

func newGaugeFunc(opts *GaugeOpts, function func() float64, v semver.Version) GaugeFunc {
	g := NewGauge(opts)

	if !g.Create(&v) {
		return nil
	}

	return prometheus.NewGaugeFunc(g.GaugeOpts.toPromGaugeOpts(), function)
}

// NewGaugeFunc creates a new GaugeFunc based on the provided GaugeOpts. The
// value reported is determined by calling the given function from within the
// Write method. Take into account that metric collection may happen
// concurrently. If that results in concurrent calls to Write, like in the case
// where a GaugeFunc is directly registered with Prometheus, the provided
// function must be concurrency-safe.
func NewGaugeFunc(opts *GaugeOpts, function func() float64) GaugeFunc {
	v := parseVersion(version.Get())

	return newGaugeFunc(opts, function, v)
}

// WithContext returns wrapped GaugeVec with context
func (v *GaugeVec) WithContext(ctx context.Context) *GaugeVecWithContext {
	return &GaugeVecWithContext{
		ctx:      ctx,
		GaugeVec: v,
	}
}

func (v *GaugeVec) InterfaceWithContext(ctx context.Context) GaugeVecMetric {
	return v.WithContext(ctx)
}

// GaugeVecWithContext is the wrapper of GaugeVec with context.
type GaugeVecWithContext struct {
	*GaugeVec
	ctx context.Context
}

// WithLabelValues is the wrapper of GaugeVec.WithLabelValues.
func (vc *GaugeVecWithContext) WithLabelValues(lvs ...string) GaugeMetric {
	return vc.GaugeVec.WithLabelValues(lvs...)
}

// With is the wrapper of GaugeVec.With.
func (vc *GaugeVecWithContext) With(labels map[string]string) GaugeMetric {
	return vc.GaugeVec.With(labels)
}
