/*
Copyright 2024 The Kubernetes Authors.

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

package v1

// MetricsConfiguration contains all metrics options.
type MetricsConfiguration struct {
	// ShowHiddenMetricsForVersion is the previous version for which you want to show hidden metrics.
	// Only the previous minor version is meaningful, other values will not be allowed.
	// The format is <major>.<minor>, e.g.: '1.16'.
	// The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics,
	// rather than being surprised when they are permanently removed in the release after that.
	ShowHiddenMetricsForVersion string `json:"showHiddenMetricsForVersion,omitempty"`
	// DisabledMetrics is a list of fully qualified metric names that should be disabled.
	// Disabling metrics is higher in precedence than showing hidden metrics.
	DisabledMetrics []string `json:"disabledMetrics,omitempty"`
	// AllowListMapping is the map from metric-label to value allow-list of this label.
	// The key's format is <MetricName>,<LabelName>, while its value is a list of allowed values for that label of that metric. For e.g., "metric1,label1": "v1,v2,v3".
	AllowListMapping map[string]string `json:"allowListMapping,omitempty"`
	// The path to the manifest file that contains the allow-list mapping. Provided for convenience over AllowListMapping.
	// NOTE: When this is supplied as a flag, the path resolution is relative to the current working directory.
	// However, when this is supplied as a configuration file, the path resolution is relative to the configuration file's location.
	// Components are required to pass in the resolved absolute path in such cases.
	// TODO: Check this somehow??
	AllowListMappingManifest string `json:"allowListMappingManifest,omitempty"`
}

// Ideally, we'd like to keep all accompanying code in the same package, but
// this is not possible due to backward-compatibility reasons. As such, that
// code resides in the `k8s.io/component-base/metrics` package.
