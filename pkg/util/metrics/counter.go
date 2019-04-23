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
func NewCounter(opts CounterOpts) *kubeCounter {
	// todo: handle defaulting better
	if opts.StabilityLevel == "" {
		opts.StabilityLevel = ALPHA
	}
	kc := &kubeCounter{
		CounterOpts: &opts,
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

// GetDeprecatedVersion returns a pointer to the Version or nil
func (c *kubeCounter) GetDeprecatedVersion() *semver.Version {
	return c.CounterOpts.DeprecatedVersion
}

// initializeMetric invocation creates the actual underlying Counter. Until this method is called
// our underlying counter is a no-op.
func (c *kubeCounter) initializeMetric() {
	c.CounterOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus counter.
	c.setPrometheusCounter(prometheus.NewCounter(c.CounterOpts.toPromCounterOpts()))
}

// initializeDeprecatedMetric invocation creates the actual (but deprecated) Counter. Until this method
// is called our underlying counter is a no-op.
func (c *kubeCounter) initializeDeprecatedMetric() {
	c.CounterOpts.markDeprecated()
	c.initializeMetric()
}

// kubeCounterVec is our internal representation of our wrapping struct around prometheus
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
func NewCounterVec(opts CounterOpts, labels []string) *kubeCounterVec {
	cv := &kubeCounterVec{
		CounterVec:     noopCounterVec,
		CounterOpts:    &opts,
		originalLabels: labels,
		lazyMetric:     lazyMetric{},
	}
	cv.lazyInit(cv)
	return cv
}

// GetDeprecatedVersion returns a pointer to the Version or nil
func (v *kubeCounterVec) GetDeprecatedVersion() *semver.Version {
	return v.CounterOpts.DeprecatedVersion
}

// initializeMetric invocation creates the actual underlying CounterVec. Until this method is called
// our underlying counterVec is a no-op.
func (v *kubeCounterVec) initializeMetric() {
	v.CounterVec = prometheus.NewCounterVec(v.CounterOpts.toPromCounterOpts(), v.originalLabels)
}

// initializeMetric invocation creates the actual (but deprecated) CounterVec. Until this method is called
// our underlying counterVec is a no-op.
func (v *kubeCounterVec) initializeDeprecatedMetric() {
	v.CounterOpts.markDeprecated()
	v.initializeMetric()
}

// Default Prometheus behavior actually results in the creation of a new metric
// if a metric with the unique label values is not found in the underlying stored metricMap. This
// is undesirable for us, since we want a way to turn OFF metrics which end up turning into memory
// leaks.
//
// For reference: https://github.com/prometheus/client_golang/blob/master/prometheus/counter.go#L148-L177
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
