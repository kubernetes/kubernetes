/*
Copyright 2021 The Kubernetes Authors.

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

package options

import (
	"github.com/spf13/pflag"
	"k8s.io/component-base/config/options"
	"k8s.io/component-base/config/validation"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/config"
	"k8s.io/component-base/logs/sanitization"
	"k8s.io/klog/v2"
)

// Options has klog format parameters
type Options struct {
	Config config.LoggingConfiguration
}

// NewOptions return new klog options
func NewOptions() *Options {
	return &Options{Config: config.LoggingConfiguration{Format: options.DefaultLogFormat}}
}

// Validate verifies if any unsupported flag is set
// for non-default logging format
func (o *Options) Validate() []error {
	errs := validation.ValidateLoggingConfiguration(&o.Config, field.NewPath("loggingConfiguration"))
	if len(errs) != 0 {
		return errs.ToAggregate().Errors()
	}
	return []error{}
}

// AddFlags add logging-format flag
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	options.BindLoggingFlags(&o.Config, fs)
}

// Apply set klog logger from LogFormat type
func (o *Options) Apply() {
	// if log format not exists, use nil loggr
	loggr, _ := options.LogRegistry.Get(o.Config.Format)
	klog.SetLogger(loggr)
	if o.Config.Sanitization {
		klog.SetLogFilter(&sanitization.SanitizingFilter{})
	}
}
