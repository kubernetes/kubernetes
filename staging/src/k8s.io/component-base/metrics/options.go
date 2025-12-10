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
	"regexp"

	"github.com/blang/semver/v4"
	"github.com/spf13/pflag"

	"k8s.io/component-base/version"
)

// Options has all parameters needed for exposing metrics from components
type Options struct {
	ShowHiddenMetricsForVersion string
	DisabledMetrics             []string
	AllowListMapping            map[string]string
	AllowListMappingManifest    string
}

// NewOptions returns default metrics options
func NewOptions() *Options {
	return &Options{}
}

// Validate validates metrics flags options.
func (o *Options) Validate() []error {
	if o == nil {
		return nil
	}

	var errs []error
	err := validateShowHiddenMetricsVersion(parseVersion(version.Get()), o.ShowHiddenMetricsForVersion)
	if err != nil {
		errs = append(errs, err)
	}

	if err := validateAllowMetricLabel(o.AllowListMapping); err != nil {
		errs = append(errs, err)
	}

	if len(errs) == 0 {
		return nil
	}
	return errs
}

// AddFlags adds flags for exposing component metrics.
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
			"The format of the file is the same as the flag --allow-metric-labels. "+
			"Note that the flag --allow-metric-labels will override the manifest file.")
}

// Apply applies parameters into global configuration of metrics.
func (o *Options) Apply() {
	if o == nil {
		return
	}
	if len(o.ShowHiddenMetricsForVersion) > 0 {
		SetShowHidden()
	}
	// set disabled metrics
	for _, metricName := range o.DisabledMetrics {
		SetDisabledMetric(metricName)
	}
	if o.AllowListMapping != nil {
		SetLabelAllowListFromCLI(o.AllowListMapping)
	} else if len(o.AllowListMappingManifest) > 0 {
		SetLabelAllowListFromManifest(o.AllowListMappingManifest)
	}
}

func validateShowHiddenMetricsVersion(currentVersion semver.Version, targetVersionStr string) error {
	if targetVersionStr == "" {
		return nil
	}

	validVersionStr := fmt.Sprintf("%d.%d", currentVersion.Major, currentVersion.Minor-1)
	if targetVersionStr != validVersionStr {
		return fmt.Errorf("--show-hidden-metrics-for-version must be omitted or have the value '%v'. Only the previous minor version is allowed", validVersionStr)
	}

	return nil
}

func validateAllowMetricLabel(allowListMapping map[string]string) error {
	if allowListMapping == nil {
		return nil
	}
	metricNameRegex := `[a-zA-Z_:][a-zA-Z0-9_:]*`
	labelRegex := `[a-zA-Z_][a-zA-Z0-9_]*`
	for k := range allowListMapping {
		reg := regexp.MustCompile(metricNameRegex + `,` + labelRegex)
		if reg.FindString(k) != k {
			return fmt.Errorf("--allow-metric-labels must have a list of kv pair with format `metricName,labelName=labelValue, labelValue,...`")
		}
	}
	return nil
}
