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

// Summary is our internal representation for our wrapping struct around prometheus
// summaries. Summary implements both KubeCollector and ObserverMetric
//
// DEPRECATED: as per the metrics overhaul KEP
type Summary struct {
	ObserverMetric
	*SummaryOpts
	lazyMetric
	selfCollector
}

// NewSummary returns an object which is Summary-like. However, nothing
// will be measured until the summary is registered somewhere.
//
// DEPRECATED: as per the metrics overhaul KEP
func NewSummary(opts *SummaryOpts) *Summary {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	s := &Summary{
		SummaryOpts: opts,
		lazyMetric:  lazyMetric{},
	}
	s.setPrometheusSummary(noopMetric{})
	s.lazyInit(s)
	return s
}

// setPrometheusSummary sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (s *Summary) setPrometheusSummary(summary prometheus.Summary) {
	s.ObserverMetric = summary
	s.initSelfCollection(summary)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (s *Summary) DeprecatedVersion() *semver.Version {
	return s.SummaryOpts.DeprecatedVersion
}

// initializeMetric invokes the actual prometheus.Summary object instantiation
// and stores a reference to it
func (s *Summary) initializeMetric() {
	s.SummaryOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus gauge.
	s.setPrometheusSummary(prometheus.NewSummary(s.SummaryOpts.toPromSummaryOpts()))
}

// initializeDeprecatedMetric invokes the actual prometheus.Summary object instantiation
// but modifies the Help description prior to object instantiation.
func (s *Summary) initializeDeprecatedMetric() {
	s.SummaryOpts.markDeprecated()
	s.initializeMetric()
}

// SummaryVec is the internal representation of our wrapping struct around prometheus
// summaryVecs.
//
// DEPRECATED: as per the metrics overhaul KEP
type SummaryVec struct {
	*prometheus.SummaryVec
	*SummaryOpts
	lazyMetric
	originalLabels []string
}

// NewSummaryVec returns an object which satisfies KubeCollector and wraps the
// prometheus.SummaryVec object. However, the object returned will not measure
// anything unless the collector is first registered, since the metric is lazily instantiated.
//
// DEPRECATED: as per the metrics overhaul KEP
func NewSummaryVec(opts *SummaryOpts, labels []string) *SummaryVec {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	v := &SummaryVec{
		SummaryOpts:    opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{},
	}
	v.lazyInit(v)
	return v
}

// DeprecatedVersion returns a pointer to the Version or nil
func (v *SummaryVec) DeprecatedVersion() *semver.Version {
	return v.SummaryOpts.DeprecatedVersion
}

func (v *SummaryVec) initializeMetric() {
	v.SummaryOpts.annotateStabilityLevel()
	v.SummaryVec = prometheus.NewSummaryVec(v.SummaryOpts.toPromSummaryOpts(), v.originalLabels)
}

func (v *SummaryVec) initializeDeprecatedMetric() {
	v.SummaryOpts.markDeprecated()
	v.initializeMetric()
}

// Default Prometheus behavior actually results in the creation of a new metric
// if a metric with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/summary.go#L485-L495

// WithLabelValues returns the ObserverMetric for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new ObserverMetric is created IFF the summaryVec
// has been registered to a metrics registry.
func (v *SummaryVec) WithLabelValues(lvs ...string) ObserverMetric {
	if !v.IsCreated() {
		return noop
	}
	return v.SummaryVec.WithLabelValues(lvs...)
}

// With returns the ObserverMetric for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new ObserverMetric is created IFF the summaryVec has
// been registered to a metrics registry.
func (v *SummaryVec) With(labels prometheus.Labels) ObserverMetric {
	if !v.IsCreated() {
		return noop
	}
	return v.SummaryVec.With(labels)
}
