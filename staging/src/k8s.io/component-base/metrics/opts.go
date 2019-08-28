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
	"github.com/prometheus/client_golang/prometheus"
	"sync"
	"time"
)

// KubeOpts is superset struct for prometheus.Opts. The prometheus Opts structure
// is purposefully not embedded here because that would change struct initialization
// in the manner which people are currently accustomed.
//
// Name must be set to a non-empty string. DeprecatedVersion is defined only
// if the metric for which this options applies is, in fact, deprecated.
type KubeOpts struct {
	Namespace         string
	Subsystem         string
	Name              string
	Help              string
	ConstLabels       prometheus.Labels
	DeprecatedVersion string
	deprecateOnce     sync.Once
	annotateOnce      sync.Once
	StabilityLevel    StabilityLevel
}

// StabilityLevel represents the API guarantees for a given defined metric.
type StabilityLevel string

const (
	// ALPHA metrics have no stability guarantees, as such, labels may
	// be arbitrarily added/removed and the metric may be deleted at any time.
	ALPHA StabilityLevel = "ALPHA"
	// STABLE metrics are guaranteed not be mutated and removal is governed by
	// the deprecation policy outlined in by the control plane metrics stability KEP.
	STABLE StabilityLevel = "STABLE"
)

// CounterOpts is an alias for Opts. See there for doc comments.
type CounterOpts KubeOpts

// Modify help description on the metric description.
func (o *CounterOpts) markDeprecated() {
	o.deprecateOnce.Do(func() {
		o.Help = fmt.Sprintf("(Deprecated since %v) %v", o.DeprecatedVersion, o.Help)
	})
}

// annotateStabilityLevel annotates help description on the metric description with the stability level
// of the metric
func (o *CounterOpts) annotateStabilityLevel() {
	o.annotateOnce.Do(func() {
		o.Help = fmt.Sprintf("[%v] %v", o.StabilityLevel, o.Help)
	})
}

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o *CounterOpts) toPromCounterOpts() prometheus.CounterOpts {
	return prometheus.CounterOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
	}
}

// GaugeOpts is an alias for Opts. See there for doc comments.
type GaugeOpts KubeOpts

// Modify help description on the metric description.
func (o *GaugeOpts) markDeprecated() {
	o.deprecateOnce.Do(func() {
		o.Help = fmt.Sprintf("(Deprecated since %v) %v", o.DeprecatedVersion, o.Help)
	})
}

// annotateStabilityLevel annotates help description on the metric description with the stability level
// of the metric
func (o *GaugeOpts) annotateStabilityLevel() {
	o.annotateOnce.Do(func() {
		o.Help = fmt.Sprintf("[%v] %v", o.StabilityLevel, o.Help)
	})
}

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o GaugeOpts) toPromGaugeOpts() prometheus.GaugeOpts {
	return prometheus.GaugeOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
	}
}

// HistogramOpts bundles the options for creating a Histogram metric. It is
// mandatory to set Name to a non-empty string. All other fields are optional
// and can safely be left at their zero value, although it is strongly
// encouraged to set a Help string.
type HistogramOpts struct {
	Namespace         string
	Subsystem         string
	Name              string
	Help              string
	ConstLabels       prometheus.Labels
	Buckets           []float64
	DeprecatedVersion string
	deprecateOnce     sync.Once
	annotateOnce      sync.Once
	StabilityLevel    StabilityLevel
}

// Modify help description on the metric description.
func (o *HistogramOpts) markDeprecated() {
	o.deprecateOnce.Do(func() {
		o.Help = fmt.Sprintf("(Deprecated since %v) %v", o.DeprecatedVersion, o.Help)
	})
}

// annotateStabilityLevel annotates help description on the metric description with the stability level
// of the metric
func (o *HistogramOpts) annotateStabilityLevel() {
	o.annotateOnce.Do(func() {
		o.Help = fmt.Sprintf("[%v] %v", o.StabilityLevel, o.Help)
	})
}

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o HistogramOpts) toPromHistogramOpts() prometheus.HistogramOpts {
	return prometheus.HistogramOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
		Buckets:     o.Buckets,
	}
}

// SummaryOpts bundles the options for creating a Summary metric. It is
// mandatory to set Name to a non-empty string. While all other fields are
// optional and can safely be left at their zero value, it is recommended to set
// a help string and to explicitly set the Objectives field to the desired value
// as the default value will change in the upcoming v0.10 of the library.
type SummaryOpts struct {
	Namespace         string
	Subsystem         string
	Name              string
	Help              string
	ConstLabels       prometheus.Labels
	Objectives        map[float64]float64
	MaxAge            time.Duration
	AgeBuckets        uint32
	BufCap            uint32
	DeprecatedVersion string
	deprecateOnce     sync.Once
	annotateOnce      sync.Once
	StabilityLevel    StabilityLevel
}

// Modify help description on the metric description.
func (o *SummaryOpts) markDeprecated() {
	o.deprecateOnce.Do(func() {
		o.Help = fmt.Sprintf("(Deprecated since %v) %v", o.DeprecatedVersion, o.Help)
	})
}

// annotateStabilityLevel annotates help description on the metric description with the stability level
// of the metric
func (o *SummaryOpts) annotateStabilityLevel() {
	o.annotateOnce.Do(func() {
		o.Help = fmt.Sprintf("[%v] %v", o.StabilityLevel, o.Help)
	})
}

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o SummaryOpts) toPromSummaryOpts() prometheus.SummaryOpts {
	return prometheus.SummaryOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
		Objectives:  o.Objectives,
		MaxAge:      o.MaxAge,
		AgeBuckets:  o.AgeBuckets,
		BufCap:      o.BufCap,
	}
}
