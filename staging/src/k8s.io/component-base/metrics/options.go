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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/version"
	"k8s.io/klog/v2"

	"go.yaml.in/yaml/v2"
	"k8s.io/component-base/metrics/api/v1"
)

var (
	disabledMetricsLock sync.RWMutex
	disabledMetrics     = map[string]struct{}{}
	showHiddenOnce      sync.Once
	showHidden          atomic.Bool
)

var (
	disabledMetricsTotal = NewCounter(
		&CounterOpts{
			Name:           "disabled_metrics_total",
			Help:           "The count of disabled metrics.",
			StabilityLevel: BETA,
		},
	)

	hiddenMetricsTotal = NewCounter(
		&CounterOpts{
			Name:           "hidden_metrics_total",
			Help:           "The count of hidden metrics.",
			StabilityLevel: BETA,
		},
	)

	cardinalityEnforcementUnexpectedCategorizationsTotal = NewCounter(
		&CounterOpts{
			Name:           "cardinality_enforcement_unexpected_categorizations_total",
			Help:           "The count of unexpected categorizations during cardinality enforcement.",
			StabilityLevel: ALPHA,
		},
	)
)

// Options has all parameters needed for exposing metrics from components
type Options struct {
	// Configuration serialization is omitted here since the parent is never expected to be embedded.
	v1.MetricsConfiguration `json:"-"`
}

// NewOptions returns default metrics options
func NewOptions() *Options {
	return &Options{}
}

// AddFlags adds flags for exposing component metrics.
// This won't be called in embedded instances within component configurations.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}
	fs.StringVar(&o.ShowHiddenMetricsForVersion, "show-hidden-metrics-for-version", o.ShowHiddenMetricsForVersion,
		"The previous version for which you want to show hidden metrics. "+
			"Only the previous minor version is meaningful, other values will not be allowed. "+
			"The format is <major>.<minor>, e.g.: '1.16'. "+
			"The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics, "+
			"rather than being surprised when they are permanently removed in the release after that.")
	fs.StringSliceVar(&o.DisabledMetrics,
		"disabled-metrics",
		o.DisabledMetrics,
		"This flag provides an escape hatch for misbehaving metrics. "+
			"You must provide the fully qualified metric name in order to disable it. "+
			"Disclaimer: disabling metrics is higher in precedence than showing hidden metrics.")
	fs.StringToStringVar(&o.AllowListMapping, "allow-metric-labels", o.AllowListMapping,
		"The map from metric-label to value allow-list of this label. The key's format is <MetricName>,<LabelName>. "+
			"The value's format is <allowed_value>,<allowed_value>..."+
			"e.g. metric1,label1='v1,v2,v3', metric1,label2='v1,v2,v3' metric2,label1='v1,v2,v3'.")
	fs.StringVar(&o.AllowListMappingManifest, "allow-metric-labels-manifest", o.AllowListMappingManifest,
		"The path to the manifest file that contains the allow-list mapping. "+
			"The format of the file is the same as the flag --allow-metric-labels, i.e., \n"+
			"allowListMapping:\n  \"metric1,label1\": \"value11,value12\"\n  \"metric2,label2\": \"\"\n"+
			"Note that the flag --allow-metric-labels will override the manifest file.")
}

// SetShowHidden will enable showing hidden metrics. This will no-opt
// after the initial call
func SetShowHidden() {
	showHiddenOnce.Do(func() {
		showHidden.Store(true)

		// re-register collectors that has been hidden in phase of last registry.
		for _, r := range registries {
			r.enableHiddenCollectors()
			r.enableHiddenStableCollectors()
		}
	})
}

// ShouldShowHidden returns whether showing hidden deprecated metrics is enabled.
// While the primary use case for this is internal (to determine registration behavior) this can also be used to introspect.
func ShouldShowHidden() bool {
	return showHidden.Load()
}

// SetDisabledMetric will disable a metric by name.
// This will also increment the disabled metrics counter.
// Note that this is a no-op if the metric is already disabled.
func SetDisabledMetrics(names []string) {
	for _, name := range names {
		func(name string) {
			// An empty metric name is not a valid Prometheus metric.
			if name == "" {
				klog.Warningf("Attempted to disable an empty metric name, ignoring.")
				return
			}
			disabledMetricsLock.Lock()
			defer disabledMetricsLock.Unlock()
			if _, ok := disabledMetrics[name]; !ok {
				disabledMetrics[name] = struct{}{}
				disabledMetricsTotal.Inc()
			}
		}(name)
	}
}

type MetricLabelAllowList struct {
	labelToAllowList map[string]sets.Set[string]
}

func (allowList *MetricLabelAllowList) ConstrainToAllowedList(labelNameList, labelValueList []string) {
	for index, value := range labelValueList {
		name := labelNameList[index]
		if allowValues, ok := allowList.labelToAllowList[name]; ok {
			if !allowValues.Has(value) {
				labelValueList[index] = "unexpected"
				cardinalityEnforcementUnexpectedCategorizationsTotal.Inc()
			}
		}
	}
}

func (allowList *MetricLabelAllowList) ConstrainLabelMap(labels map[string]string) {
	for name, value := range labels {
		if allowValues, ok := allowList.labelToAllowList[name]; ok {
			if !allowValues.Has(value) {
				labels[name] = "unexpected"
				cardinalityEnforcementUnexpectedCategorizationsTotal.Inc()
			}
		}
	}
}

func SetLabelAllowList(allowListMapping map[string]string) {
	if len(allowListMapping) == 0 {
		klog.Errorf("empty allow-list mapping supplied, ignoring.")
		return
	}

	allowListLock.Lock()
	defer allowListLock.Unlock()
	for metricLabelName, labelValues := range allowListMapping {
		metricName := strings.Split(metricLabelName, ",")[0]
		labelName := strings.Split(metricLabelName, ",")[1]
		valueSet := sets.New[string](strings.Split(labelValues, ",")...)

		allowList, ok := labelValueAllowLists[metricName]
		if ok {
			allowList.labelToAllowList[labelName] = valueSet
		} else {
			labelToAllowList := make(map[string]sets.Set[string])
			labelToAllowList[labelName] = valueSet
			labelValueAllowLists[metricName] = &MetricLabelAllowList{
				labelToAllowList,
			}
		}
	}
}

func SetLabelAllowListFromManifest(manifest string) {
	if manifest == "" {
		klog.Errorf("The manifest file is empty, ignoring.")
		return
	}

	data, err := os.ReadFile(filepath.Clean(manifest))
	if err != nil {
		klog.Errorf("Failed to read allow list manifest: %v", err)
		return
	}
	allowListMapping := make(map[string]string)
	err = yaml.Unmarshal(data, &allowListMapping)
	if err != nil {
		klog.Errorf("Failed to parse allow list manifest: %v", err)
		return
	}
	SetLabelAllowList(allowListMapping)
}

// Apply applies a MetricsConfiguration into global configuration of metrics.
func Apply(c *v1.MetricsConfiguration) {
	if c == nil {
		return
	}

	if len(c.ShowHiddenMetricsForVersion) > 0 {
		SetShowHidden()
	}
	SetDisabledMetrics(c.DisabledMetrics)
	if c.AllowListMapping != nil {
		SetLabelAllowList(c.AllowListMapping)
	} else {
		SetLabelAllowListFromManifest(c.AllowListMappingManifest)
	}
}

// Apply applies parameters into global configuration of metrics.
func (o *Options) Apply() {
	if o == nil {
		return
	}

	Apply(&o.MetricsConfiguration)
}

// ValidateShowHiddenMetricsVersion checks invalid version for which show hidden metrics.
// TODO: This is kept here for backward compatibility in Kubelet (as metrics configuration fields were exposed on an individual basis earlier).
// TODO: Revisit this after Kubelet supports the new metrics configuration API.
func ValidateShowHiddenMetricsVersion(v string) []error {
	err := v1.ValidateShowHiddenMetricsVersionForKubeletBackwardCompatOnly(parseVersion(version.Get()), v)
	if err != nil {
		return []error{err}
	}

	return nil
}

// Validate validates metrics flags options.
func (o *Options) Validate() []error {
	if o == nil {
		return nil
	}

	return v1.Validate(&o.MetricsConfiguration, parseVersion(version.Get()), field.NewPath("metrics")).ToAggregate().Errors()
}
