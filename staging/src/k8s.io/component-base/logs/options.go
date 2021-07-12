/*
Copyright 2020 The Kubernetes Authors.

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

package logs

import (
	"github.com/spf13/pflag"

	"k8s.io/klog/v2"

	"k8s.io/component-base/config"
	"k8s.io/component-base/config/v1alpha1"
	"k8s.io/component-base/logs/sanitization"
)

// Options has klog format parameters
type Options struct {
	Config config.LoggingConfiguration
}

// NewOptions return new klog options
func NewOptions() *Options {
	c := v1alpha1.LoggingConfiguration{}
	v1alpha1.RecommendedLoggingConfiguration(&c)
	o := &Options{}
	v1alpha1.Convert_v1alpha1_LoggingConfiguration_To_config_LoggingConfiguration(&c, &o.Config, nil)
	return o
}

// Validate verifies if any unsupported flag is set
// for non-default logging format
func (o *Options) Validate() []error {
	errs := ValidateLoggingConfiguration(&o.Config, nil)
	if len(errs) != 0 {
		return errs.ToAggregate().Errors()
	}
	return nil
}

// AddFlags add logging-format flag
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	BindLoggingFlags(&o.Config, fs)
}

// Apply set klog logger from LogFormat type
func (o *Options) Apply() {
	// if log format not exists, use nil loggr
	loggr, _ := LogRegistry.Get(o.Config.Format)
	klog.SetLogger(loggr)
	if o.Config.Sanitization {
		klog.SetLogFilter(&sanitization.SanitizingFilter{})
	}
}
