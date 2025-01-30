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
	"sync"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	promext "k8s.io/component-base/metrics/prometheusextension"
	"k8s.io/klog/v2"
)

/*
kubeCollector extends the prometheus.Collector interface to allow customization of the metric
registration process. Defer metric initialization until Create() is called, which then
delegates to the underlying metric's initializeMetric or initializeDeprecatedMetric
method call depending on whether the metric is deprecated or not.
*/
type kubeCollector interface {
	Collector
	lazyKubeMetric
	DeprecatedVersion() *semver.Version
	// Each collector metric should provide an initialization function
	// for both deprecated and non-deprecated variants of a metric. This
	// is necessary since metric instantiation will be deferred
	// until the metric is actually registered somewhere.
	initializeMetric()
	initializeDeprecatedMetric()
}

/*
lazyKubeMetric defines our metric registration interface. lazyKubeMetric objects are expected
to lazily instantiate metrics (i.e defer metric instantiation until when
the Create() function is explicitly called).
*/
type lazyKubeMetric interface {
	Create(*semver.Version) bool
	IsCreated() bool
	IsHidden() bool
	IsDeprecated() bool
}

/*
lazyMetric implements lazyKubeMetric. A lazy metric is lazy because it waits until metric
registration time before instantiation. Add it as an anonymous field to a struct that
implements kubeCollector to get deferred registration behavior. You must call lazyInit
with the kubeCollector itself as an argument.
*/
type lazyMetric struct {
	fqName              string
	isDeprecated        bool
	isHidden            bool
	isCreated           bool
	createLock          sync.RWMutex
	markDeprecationOnce sync.Once
	createOnce          sync.Once
	self                kubeCollector
	stabilityLevel      StabilityLevel
}

func (r *lazyMetric) IsCreated() bool {
	r.createLock.RLock()
	defer r.createLock.RUnlock()
	return r.isCreated
}

// lazyInit provides the lazyMetric with a reference to the kubeCollector it is supposed
// to allow lazy initialization for. It should be invoked in the factory function which creates new
// kubeCollector type objects.
func (r *lazyMetric) lazyInit(self kubeCollector, fqName string) {
	r.fqName = fqName
	r.self = self
}

// preprocessMetric figures out whether the lazy metric should be hidden or not.
// This method takes a Version argument which should be the version of the binary in which
// this code is currently being executed. A metric can be hidden under two conditions:
//  1. if the metric is deprecated and is outside the grace period (i.e. has been
//     deprecated for more than one release
//  2. if the metric is manually disabled via a CLI flag.
//
// Disclaimer:  disabling a metric via a CLI flag has higher precedence than
// deprecation and will override show-hidden-metrics for the explicitly
// disabled metric.
func (r *lazyMetric) preprocessMetric(version semver.Version) {
	disabledMetricsLock.RLock()
	defer disabledMetricsLock.RUnlock()
	// disabling metrics is higher in precedence than showing hidden metrics
	if _, ok := disabledMetrics[r.fqName]; ok {
		r.isHidden = true
		return
	}
	selfVersion := r.self.DeprecatedVersion()
	if selfVersion == nil {
		return
	}
	r.markDeprecationOnce.Do(func() {
		if selfVersion.LTE(version) {
			r.isDeprecated = true
		}

		if ShouldShowHidden() {
			klog.Warningf("Hidden metrics (%s) have been manually overridden, showing this very deprecated metric.", r.fqName)
			return
		}
		if shouldHide(&version, selfVersion) {
			// TODO(RainbowMango): Remove this log temporarily. https://github.com/kubernetes/kubernetes/issues/85369
			// klog.Warningf("This metric has been deprecated for more than one release, hiding.")
			r.isHidden = true
		}
	})
}

func (r *lazyMetric) IsHidden() bool {
	return r.isHidden
}

func (r *lazyMetric) IsDeprecated() bool {
	return r.isDeprecated
}

// Create forces the initialization of metric which has been deferred until
// the point at which this method is invoked. This method will determine whether
// the metric is deprecated or hidden, no-opting if the metric should be considered
// hidden. Furthermore, this function no-opts and returns true if metric is already
// created.
func (r *lazyMetric) Create(version *semver.Version) bool {
	if version != nil {
		r.preprocessMetric(*version)
	}
	// let's not create if this metric is slated to be hidden
	if r.IsHidden() {
		return false
	}

	r.createOnce.Do(func() {
		r.createLock.Lock()
		defer r.createLock.Unlock()
		r.isCreated = true
		if r.IsDeprecated() {
			r.self.initializeDeprecatedMetric()
		} else {
			r.self.initializeMetric()
		}
	})
	sl := r.stabilityLevel
	deprecatedV := r.self.DeprecatedVersion()
	dv := ""
	if deprecatedV != nil {
		dv = deprecatedV.String()
	}
	registeredMetricsTotal.WithLabelValues(string(sl), dv).Inc()
	return r.IsCreated()
}

// ClearState will clear all the states marked by Create.
// It intends to be used for re-register a hidden metric.
func (r *lazyMetric) ClearState() {
	r.createLock.Lock()
	defer r.createLock.Unlock()

	r.isDeprecated = false
	r.isHidden = false
	r.isCreated = false
	r.markDeprecationOnce = sync.Once{}
	r.createOnce = sync.Once{}
}

// FQName returns the fully-qualified metric name of the collector.
func (r *lazyMetric) FQName() string {
	return r.fqName
}

/*
This code is directly lifted from the prometheus codebase. It's a convenience struct which
allows you satisfy the Collector interface automatically if you already satisfy the Metric interface.

For reference: https://github.com/prometheus/client_golang/blob/v0.9.2/prometheus/collector.go#L98-L120
*/
type selfCollector struct {
	metric prometheus.Metric
}

func (c *selfCollector) initSelfCollection(m prometheus.Metric) {
	c.metric = m
}

func (c *selfCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.metric.Desc()
}

func (c *selfCollector) Collect(ch chan<- prometheus.Metric) {
	ch <- c.metric
}

// metricWithExemplar is an interface that knows how to attach an exemplar to certain supported metric types.
type metricWithExemplar interface {
	withExemplar(v float64)
}

// no-op vecs for convenience
var noopCounterVec = &prometheus.CounterVec{}
var noopHistogramVec = &prometheus.HistogramVec{}
var noopTimingHistogramVec = &promext.TimingHistogramVec{}
var noopGaugeVec = &prometheus.GaugeVec{}

// just use a convenience struct for all the no-ops
var noop = &noopMetric{}

type noopMetric struct{}

func (noopMetric) Inc()                              {}
func (noopMetric) Add(float64)                       {}
func (noopMetric) Dec()                              {}
func (noopMetric) Set(float64)                       {}
func (noopMetric) Sub(float64)                       {}
func (noopMetric) Observe(float64)                   {}
func (noopMetric) ObserveWithWeight(float64, uint64) {}
func (noopMetric) SetToCurrentTime()                 {}
func (noopMetric) Desc() *prometheus.Desc            { return nil }
func (noopMetric) Write(*dto.Metric) error           { return nil }
func (noopMetric) Describe(chan<- *prometheus.Desc)  {}
func (noopMetric) Collect(chan<- prometheus.Metric)  {}
