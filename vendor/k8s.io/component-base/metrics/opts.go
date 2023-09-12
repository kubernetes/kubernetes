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
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/util/sets"
	promext "k8s.io/component-base/metrics/prometheusextension"
)

var (
	labelValueAllowLists = map[string]*MetricLabelAllowList{}
	allowListLock        sync.RWMutex
)

// KubeOpts is superset struct for prometheus.Opts. The prometheus Opts structure
// is purposefully not embedded here because that would change struct initialization
// in the manner which people are currently accustomed.
//
// Name must be set to a non-empty string. DeprecatedVersion is defined only
// if the metric for which this options applies is, in fact, deprecated.
type KubeOpts struct {
	Namespace            string
	Subsystem            string
	Name                 string
	Help                 string
	ConstLabels          map[string]string
	DeprecatedVersion    string
	deprecateOnce        sync.Once
	annotateOnce         sync.Once
	StabilityLevel       StabilityLevel
	LabelValueAllowLists *MetricLabelAllowList
}

// BuildFQName joins the given three name components by "_". Empty name
// components are ignored. If the name parameter itself is empty, an empty
// string is returned, no matter what. Metric implementations included in this
// library use this function internally to generate the fully-qualified metric
// name from the name component in their Opts. Users of the library will only
// need this function if they implement their own Metric or instantiate a Desc
// (with NewDesc) directly.
func BuildFQName(namespace, subsystem, name string) string {
	return prometheus.BuildFQName(namespace, subsystem, name)
}

// StabilityLevel represents the API guarantees for a given defined metric.
type StabilityLevel string

const (
	// INTERNAL metrics have no stability guarantees, as such, labels may
	// be arbitrarily added/removed and the metric may be deleted at any time.
	INTERNAL StabilityLevel = "INTERNAL"
	// ALPHA metrics have no stability guarantees, as such, labels may
	// be arbitrarily added/removed and the metric may be deleted at any time.
	ALPHA StabilityLevel = "ALPHA"
	// BETA metrics are governed by the deprecation policy outlined in by
	// the control plane metrics stability KEP.
	BETA StabilityLevel = "BETA"
	// STABLE metrics are guaranteed not be mutated and removal is governed by
	// the deprecation policy outlined in by the control plane metrics stability KEP.
	STABLE StabilityLevel = "STABLE"
)

// setDefaults takes 'ALPHA' in case of empty.
func (sl *StabilityLevel) setDefaults() {
	switch *sl {
	case "":
		*sl = ALPHA
	default:
		// no-op, since we have a StabilityLevel already
	}
}

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
func (o *GaugeOpts) toPromGaugeOpts() prometheus.GaugeOpts {
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
	Namespace            string
	Subsystem            string
	Name                 string
	Help                 string
	ConstLabels          map[string]string
	Buckets              []float64
	DeprecatedVersion    string
	deprecateOnce        sync.Once
	annotateOnce         sync.Once
	StabilityLevel       StabilityLevel
	LabelValueAllowLists *MetricLabelAllowList
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
func (o *HistogramOpts) toPromHistogramOpts() prometheus.HistogramOpts {
	return prometheus.HistogramOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
		Buckets:     o.Buckets,
	}
}

// TimingHistogramOpts bundles the options for creating a TimingHistogram metric. It is
// mandatory to set Name to a non-empty string. All other fields are optional
// and can safely be left at their zero value, although it is strongly
// encouraged to set a Help string.
type TimingHistogramOpts struct {
	Namespace            string
	Subsystem            string
	Name                 string
	Help                 string
	ConstLabels          map[string]string
	Buckets              []float64
	InitialValue         float64
	DeprecatedVersion    string
	deprecateOnce        sync.Once
	annotateOnce         sync.Once
	StabilityLevel       StabilityLevel
	LabelValueAllowLists *MetricLabelAllowList
}

// Modify help description on the metric description.
func (o *TimingHistogramOpts) markDeprecated() {
	o.deprecateOnce.Do(func() {
		o.Help = fmt.Sprintf("(Deprecated since %v) %v", o.DeprecatedVersion, o.Help)
	})
}

// annotateStabilityLevel annotates help description on the metric description with the stability level
// of the metric
func (o *TimingHistogramOpts) annotateStabilityLevel() {
	o.annotateOnce.Do(func() {
		o.Help = fmt.Sprintf("[%v] %v", o.StabilityLevel, o.Help)
	})
}

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o *TimingHistogramOpts) toPromHistogramOpts() promext.TimingHistogramOpts {
	return promext.TimingHistogramOpts{
		Namespace:    o.Namespace,
		Subsystem:    o.Subsystem,
		Name:         o.Name,
		Help:         o.Help,
		ConstLabels:  o.ConstLabels,
		Buckets:      o.Buckets,
		InitialValue: o.InitialValue,
	}
}

// SummaryOpts bundles the options for creating a Summary metric. It is
// mandatory to set Name to a non-empty string. While all other fields are
// optional and can safely be left at their zero value, it is recommended to set
// a help string and to explicitly set the Objectives field to the desired value
// as the default value will change in the upcoming v0.10 of the library.
type SummaryOpts struct {
	Namespace            string
	Subsystem            string
	Name                 string
	Help                 string
	ConstLabels          map[string]string
	Objectives           map[float64]float64
	MaxAge               time.Duration
	AgeBuckets           uint32
	BufCap               uint32
	DeprecatedVersion    string
	deprecateOnce        sync.Once
	annotateOnce         sync.Once
	StabilityLevel       StabilityLevel
	LabelValueAllowLists *MetricLabelAllowList
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

// Deprecated: DefObjectives will not be used as the default objectives in
// v1.0.0 of the library. The default Summary will have no quantiles then.
var (
	defObjectives = map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
)

// convenience function to allow easy transformation to the prometheus
// counterpart. This will do more once we have a proper label abstraction
func (o *SummaryOpts) toPromSummaryOpts() prometheus.SummaryOpts {
	// we need to retain existing quantile behavior for backwards compatibility,
	// so let's do what prometheus used to do prior to v1.
	objectives := o.Objectives
	if objectives == nil {
		objectives = defObjectives
	}
	return prometheus.SummaryOpts{
		Namespace:   o.Namespace,
		Subsystem:   o.Subsystem,
		Name:        o.Name,
		Help:        o.Help,
		ConstLabels: o.ConstLabels,
		Objectives:  objectives,
		MaxAge:      o.MaxAge,
		AgeBuckets:  o.AgeBuckets,
		BufCap:      o.BufCap,
	}
}

type MetricLabelAllowList struct {
	labelToAllowList map[string]sets.String
}

func (allowList *MetricLabelAllowList) ConstrainToAllowedList(labelNameList, labelValueList []string) {
	for index, value := range labelValueList {
		name := labelNameList[index]
		if allowValues, ok := allowList.labelToAllowList[name]; ok {
			if !allowValues.Has(value) {
				labelValueList[index] = "unexpected"
			}
		}
	}
}

func (allowList *MetricLabelAllowList) ConstrainLabelMap(labels map[string]string) {
	for name, value := range labels {
		if allowValues, ok := allowList.labelToAllowList[name]; ok {
			if !allowValues.Has(value) {
				labels[name] = "unexpected"
			}
		}
	}
}

func SetLabelAllowListFromCLI(allowListMapping map[string]string) {
	allowListLock.Lock()
	defer allowListLock.Unlock()
	for metricLabelName, labelValues := range allowListMapping {
		metricName := strings.Split(metricLabelName, ",")[0]
		labelName := strings.Split(metricLabelName, ",")[1]
		valueSet := sets.NewString(strings.Split(labelValues, ",")...)

		allowList, ok := labelValueAllowLists[metricName]
		if ok {
			allowList.labelToAllowList[labelName] = valueSet
		} else {
			labelToAllowList := make(map[string]sets.String)
			labelToAllowList[labelName] = valueSet
			labelValueAllowLists[metricName] = &MetricLabelAllowList{
				labelToAllowList,
			}
		}
	}
}
