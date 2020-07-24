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

	"github.com/blang/semver"
	"github.com/spf13/pflag"

	"k8s.io/component-base/version"
)

// Options has all parameters needed for exposing metrics from components
type Options struct {
	ShowHiddenMetricsForVersion string
}

// NewOptions returns default metrics options
func NewOptions() *Options {
	return &Options{}
}

// Validate validates metrics flags options.
func (o *Options) Validate() []error {
	err := validateShowHiddenMetricsVersion(parseVersion(version.Get()), o.ShowHiddenMetricsForVersion)
	if err != nil {
		return []error{err}
	}

	return nil
}

// AddFlags adds flags for exposing component metrics.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	if o != nil {
		o = NewOptions()
	}
	fs.StringVar(&o.ShowHiddenMetricsForVersion, "show-hidden-metrics-for-version", o.ShowHiddenMetricsForVersion,
		"The previous version for which you want to show hidden metrics. "+
			"Only the previous minor version is meaningful, other values will not be allowed. "+
			"The format is <major>.<minor>, e.g.: '1.16'. "+
			"The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics, "+
			"rather than being surprised when they are permanently removed in the release after that.")
}

// Apply applies parameters into global configuration of metrics.
func (o *Options) Apply() {
	if o != nil && len(o.ShowHiddenMetricsForVersion) > 0 {
		SetShowHidden()
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
