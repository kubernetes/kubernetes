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
	"fmt"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
)

// StableCollector extends the prometheus.Collector interface to allow customization of the
// metric registration process, it's especially intend to be used in scenario of custom collector.
type StableCollector interface {
	prometheus.Collector

	// DescribeWithStability sends the super-set of all possible metrics.Desc collected
	// by this StableCollector to the provided channel.
	DescribeWithStability(chan<- *Desc)

	// CollectWithStability sends each collected metrics.Metric via the provide channel.
	CollectWithStability(chan<- Metric)

	// Create will initialize all Desc and it intends to be called by registry.
	Create(version *semver.Version, self StableCollector) bool

	// ClearState will clear all the states marked by Create.
	ClearState()

	// HiddenMetrics tells the list of hidden metrics with fqName.
	HiddenMetrics() []string
}

// BaseStableCollector which implements almost all methods defined by StableCollector
// is a convenient assistant for custom collectors.
// It is recommended to inherit BaseStableCollector when implementing custom collectors.
type BaseStableCollector struct {
	descriptors  map[string]*Desc // stores all descriptors by pair<fqName, Desc>, these are collected from DescribeWithStability().
	registerable map[string]*Desc // stores registerable descriptors by pair<fqName, Desc>, is a subset of descriptors.
	hidden       map[string]*Desc // stores hidden descriptors by pair<fqName, Desc>, is a subset of descriptors.
	self         StableCollector
}

// DescribeWithStability sends all descriptors to the provided channel.
// Every custom collector should over-write this method.
func (bsc *BaseStableCollector) DescribeWithStability(ch chan<- *Desc) {
	panic(fmt.Errorf("custom collector should over-write DescribeWithStability method"))
}

// Describe sends all descriptors to the provided channel.
// It intended to be called by prometheus registry.
func (bsc *BaseStableCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range bsc.registerable {
		ch <- d.toPrometheusDesc()
	}
}

// CollectWithStability sends all metrics to the provided channel.
// Every custom collector should over-write this method.
func (bsc *BaseStableCollector) CollectWithStability(ch chan<- Metric) {
	panic(fmt.Errorf("custom collector should over-write CollectWithStability method"))
}

// Collect is called by the Prometheus registry when collecting metrics.
func (bsc *BaseStableCollector) Collect(ch chan<- prometheus.Metric) {
	mch := make(chan Metric)

	go func() {
		bsc.self.CollectWithStability(mch)
		close(mch)
	}()

	for m := range mch {
		// nil Metric usually means hidden metrics
		if m == nil {
			continue
		}

		ch <- prometheus.Metric(m)
	}
}

func (bsc *BaseStableCollector) add(d *Desc) {
	if len(d.fqName) == 0 {
		panic("nameless metrics will be not allowed")
	}

	if bsc.descriptors == nil {
		bsc.descriptors = make(map[string]*Desc)
	}

	if _, exist := bsc.descriptors[d.fqName]; exist {
		panic(fmt.Sprintf("duplicate metrics (%s) will be not allowed", d.fqName))
	}

	bsc.descriptors[d.fqName] = d
}

// Init intends to be called by registry.
func (bsc *BaseStableCollector) init(self StableCollector) {
	bsc.self = self

	dch := make(chan *Desc)

	// collect all possible descriptions from custom side
	go func() {
		bsc.self.DescribeWithStability(dch)
		close(dch)
	}()

	for d := range dch {
		bsc.add(d)
	}
}

func (bsc *BaseStableCollector) trackRegistrableDescriptor(d *Desc) {
	if bsc.registerable == nil {
		bsc.registerable = make(map[string]*Desc)
	}

	bsc.registerable[d.fqName] = d
}

func (bsc *BaseStableCollector) trackHiddenDescriptor(d *Desc) {
	if bsc.hidden == nil {
		bsc.hidden = make(map[string]*Desc)
	}

	bsc.hidden[d.fqName] = d
}

// Create intends to be called by registry.
// Create will return true as long as there is one or more metrics not be hidden.
// Otherwise return false, that means the whole collector will be ignored by registry.
func (bsc *BaseStableCollector) Create(version *semver.Version, self StableCollector) bool {
	bsc.init(self)

	for _, d := range bsc.descriptors {
		d.create(version)
		if d.IsHidden() {
			bsc.trackHiddenDescriptor(d)
		} else {
			bsc.trackRegistrableDescriptor(d)
		}
	}

	if len(bsc.registerable) > 0 {
		return true
	}

	return false
}

// ClearState will clear all the states marked by Create.
// It intends to be used for re-register a hidden metric.
func (bsc *BaseStableCollector) ClearState() {
	for _, d := range bsc.descriptors {
		d.ClearState()
	}

	bsc.descriptors = nil
	bsc.registerable = nil
	bsc.hidden = nil
	bsc.self = nil
}

// HiddenMetrics tells the list of hidden metrics with fqName.
func (bsc *BaseStableCollector) HiddenMetrics() (fqNames []string) {
	for i := range bsc.hidden {
		fqNames = append(fqNames, bsc.hidden[i].fqName)
	}
	return
}

// Check if our BaseStableCollector implements necessary interface
var _ StableCollector = &BaseStableCollector{}
