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
	dto "github.com/prometheus/client_model/go"
	"k8s.io/klog"
	"sync"
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
	isDeprecated        bool
	isHidden            bool
	isCreated           bool
	createLock          sync.RWMutex
	markDeprecationOnce sync.Once
	createOnce          sync.Once
	self                kubeCollector
}

func (r *lazyMetric) IsCreated() bool {
	r.createLock.RLock()
	defer r.createLock.RUnlock()
	return r.isCreated
}

// lazyInit provides the lazyMetric with a reference to the kubeCollector it is supposed
// to allow lazy initialization for. It should be invoked in the factory function which creates new
// kubeCollector type objects.
func (r *lazyMetric) lazyInit(self kubeCollector) {
	r.self = self
}

// determineDeprecationStatus figures out whether the lazy metric should be deprecated or not.
// This method takes a Version argument which should be the version of the binary in which
// this code is currently being executed.
func (r *lazyMetric) determineDeprecationStatus(version semver.Version) {
	selfVersion := r.self.DeprecatedVersion()
	if selfVersion == nil {
		return
	}
	r.markDeprecationOnce.Do(func() {
		if selfVersion.LTE(version) {
			r.isDeprecated = true
		}
		if ShouldShowHidden() {
			klog.Warningf("Hidden metrics have been manually overridden, showing this very deprecated metric.")
			return
		}
		if selfVersion.LT(version) {
			klog.Warningf("This metric has been deprecated for more than one release, hiding.")
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
		r.determineDeprecationStatus(*version)
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
	return r.IsCreated()
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

// no-op vecs for convenience
var noopCounterVec = &prometheus.CounterVec{}
var noopHistogramVec = &prometheus.HistogramVec{}
var noopSummaryVec = &prometheus.SummaryVec{}
var noopGaugeVec = &prometheus.GaugeVec{}
var noopObserverVec = &noopObserverVector{}

// just use a convenience struct for all the no-ops
var noop = &noopMetric{}

type noopMetric struct{}

func (noopMetric) Inc()                             {}
func (noopMetric) Add(float64)                      {}
func (noopMetric) Dec()                             {}
func (noopMetric) Set(float64)                      {}
func (noopMetric) Sub(float64)                      {}
func (noopMetric) Observe(float64)                  {}
func (noopMetric) SetToCurrentTime()                {}
func (noopMetric) Desc() *prometheus.Desc           { return nil }
func (noopMetric) Write(*dto.Metric) error          { return nil }
func (noopMetric) Describe(chan<- *prometheus.Desc) {}
func (noopMetric) Collect(chan<- prometheus.Metric) {}

type noopObserverVector struct{}

func (noopObserverVector) GetMetricWith(prometheus.Labels) (prometheus.Observer, error) {
	return noop, nil
}
func (noopObserverVector) GetMetricWithLabelValues(...string) (prometheus.Observer, error) {
	return noop, nil
}
func (noopObserverVector) With(prometheus.Labels) prometheus.Observer    { return noop }
func (noopObserverVector) WithLabelValues(...string) prometheus.Observer { return noop }
func (noopObserverVector) CurryWith(prometheus.Labels) (prometheus.ObserverVec, error) {
	return noopObserverVec, nil
}
func (noopObserverVector) MustCurryWith(prometheus.Labels) prometheus.ObserverVec {
	return noopObserverVec
}
func (noopObserverVector) Describe(chan<- *prometheus.Desc) {}
func (noopObserverVector) Collect(chan<- prometheus.Metric) {}
