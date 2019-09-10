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
	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
)

// Gauge is our internal representation for our wrapping struct around prometheus
// gauges. kubeGauge implements both kubeCollector and KubeGauge.
type Gauge struct {
	GaugeMetric
	*GaugeOpts
	lazyMetric
	selfCollector
}

// NewGauge returns an object which satisfies the kubeCollector and KubeGauge interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated.
func NewGauge(opts *GaugeOpts) *Gauge {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	kc := &Gauge{
		GaugeOpts:  opts,
		lazyMetric: lazyMetric{},
	}
	kc.setPrometheusGauge(noop)
	kc.lazyInit(kc)
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

// GaugeVec is the internal representation of our wrapping struct around prometheus
// gaugeVecs. kubeGaugeVec implements both kubeCollector and KubeGaugeVec.
type GaugeVec struct {
	*prometheus.GaugeVec
	*GaugeOpts
	lazyMetric
	originalLabels []string
}

// NewGaugeVec returns an object which satisfies the kubeCollector and KubeGaugeVec interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated.
func NewGaugeVec(opts *GaugeOpts, labels []string) *GaugeVec {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	cv := &GaugeVec{
		GaugeVec:       noopGaugeVec,
		GaugeOpts:      opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{},
	}
	cv.lazyInit(cv)
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

// Default Prometheus behavior actually results in the creation of a new metric
// if a metric with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/gauge.go#L190-L208

// WithLabelValues returns the GaugeMetric for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new GaugeMetric is created IFF the gaugeVec
// has been registered to a metrics registry.
func (v *GaugeVec) WithLabelValues(lvs ...string) GaugeMetric {
	if !v.IsCreated() {
		return noop // return no-op gauge
	}
	return v.GaugeVec.WithLabelValues(lvs...)
}

// With returns the GaugeMetric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new GaugeMetric is created IFF the gaugeVec has
// been registered to a metrics registry.
func (v *GaugeVec) With(labels prometheus.Labels) GaugeMetric {
	if !v.IsCreated() {
		return noop // return no-op gauge
	}
	return v.GaugeVec.With(labels)
}

// Delete deletes the metric where the variable labels are the same as those
// passed in as labels. It returns true if a metric was deleted.
//
// It is not an error if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc. However, such inconsistent Labels
// can never match an actual metric, so the method will always return false in
// that case.
func (v *GaugeVec) Delete(labels prometheus.Labels) bool {
	if !v.IsCreated() {
		return false // since we haven't created the metric, we haven't deleted a metric with the passed in values
	}
	return v.GaugeVec.Delete(labels)
}
