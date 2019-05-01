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

package framework

import (
	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
)

// kubeCounter is our internal representation for our wrapping struct around prometheus
// counters. kubeCounter implements both KubeCollector and KubeCounter.
type kubeCounter struct {
	KubeCounter
	*CounterOpts
	lazyMetric
	selfCollector
}

// NewCounter returns an object which satisfies the KubeCollector and KubeCounter interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated.
func NewCounter(opts *CounterOpts) *kubeCounter {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	kc := &kubeCounter{
		CounterOpts: opts,
		lazyMetric:  lazyMetric{},
	}
	kc.setPrometheusCounter(noop)
	kc.lazyInit(kc)
	return kc
}

// setPrometheusCounter sets the underlying KubeCounter object, i.e. the thing that does the measurement.
func (c *kubeCounter) setPrometheusCounter(counter prometheus.Counter) {
	c.KubeCounter = counter
	c.initSelfCollection(counter)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (c *kubeCounter) DeprecatedVersion() *semver.Version {
	return c.CounterOpts.DeprecatedVersion
}

// initializeMetric invocation creates the actual underlying Counter. Until this method is called
// the underlying counter is a no-op.
func (c *kubeCounter) initializeMetric() {
	c.CounterOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus counter.
	c.setPrometheusCounter(prometheus.NewCounter(c.CounterOpts.toPromCounterOpts()))
}

// initializeDeprecatedMetric invocation creates the actual (but deprecated) Counter. Until this method
// is called the underlying counter is a no-op.
func (c *kubeCounter) initializeDeprecatedMetric() {
	c.CounterOpts.markDeprecated()
	c.initializeMetric()
}

// kubeCounterVec is the internal representation of our wrapping struct around prometheus
// counterVecs. kubeCounterVec implements both KubeCollector and KubeCounterVec.
type kubeCounterVec struct {
	*prometheus.CounterVec
	*CounterOpts
	lazyMetric
	originalLabels []string
}

// NewCounterVec returns an object which satisfies the KubeCollector and KubeCounterVec interfaces.
// However, the object returned will not measure anything unless the collector is first
// registered, since the metric is lazily instantiated.
func NewCounterVec(opts *CounterOpts, labels []string) *kubeCounterVec {
	cv := &kubeCounterVec{
		CounterVec:     noopCounterVec,
		CounterOpts:    opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{},
	}
	cv.lazyInit(cv)
	return cv
}

// DeprecatedVersion returns a pointer to the Version or nil
func (v *kubeCounterVec) DeprecatedVersion() *semver.Version {
	return v.CounterOpts.DeprecatedVersion
}

// initializeMetric invocation creates the actual underlying CounterVec. Until this method is called
// the underlying counterVec is a no-op.
func (v *kubeCounterVec) initializeMetric() {
	v.CounterVec = prometheus.NewCounterVec(v.CounterOpts.toPromCounterOpts(), v.originalLabels)
}

// initializeDeprecatedMetric invocation creates the actual (but deprecated) CounterVec. Until this method is called
// the underlying counterVec is a no-op.
func (v *kubeCounterVec) initializeDeprecatedMetric() {
	v.CounterOpts.markDeprecated()
	v.initializeMetric()
}

// Default Prometheus behavior actually results in the creation of a new metric
// if a metric with the unique label values is not found in the underlying stored metricMap.
// This means  that if this function is called but the underlying metric is not registered
// (which means it will never be exposed externally nor consumed), the metric will exist in memory
// for perpetuity (i.e. throughout application lifecycle).
//
// For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/counter.go#L179-L197
//
// This method returns a no-op metric if the metric is not actually created/registered, avoiding that
// memory leak.
func (v *kubeCounterVec) WithLabelValues(lvs ...string) KubeCounter {
	if !v.IsCreated() {
		return noop // return no-op counter
	}
	return v.CounterVec.WithLabelValues(lvs...)
}

func (v *kubeCounterVec) With(labels prometheus.Labels) KubeCounter {
	if !v.IsCreated() {
		return noop // return no-op counter
	}
	return v.CounterVec.With(labels)
}
