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
	"fmt"

	"github.com/spf13/pflag"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/component-base/config"
	"k8s.io/component-base/config/v1alpha1"
	"k8s.io/component-base/logs/registry"
	"k8s.io/component-base/logs/sanitization"
	"k8s.io/klog/v2"
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

// ValidateAndApply combines validation and application of the logging configuration.
// This should be invoked as early as possible because then the rest of the program
// startup (including validation of other options) will already run with the final
// logging configuration.
func (o *Options) ValidateAndApply() error {
	errs := o.validate()
	if len(errs) > 0 {
		return utilerrors.NewAggregate(errs)
	}
	o.apply()
	return nil
}

// validate verifies if any unsupported flag is set
// for non-default logging format
func (o *Options) validate() []error {
	errs := ValidateLoggingConfiguration(&o.Config, nil)
	if len(errs) != 0 {
		return errs.ToAggregate().Errors()
	}
	return nil
}

// AddFlags add logging-format flag.
//
// Programs using LoggingConfiguration must use SkipLoggingConfigurationFlags
// when calling AddFlags to avoid the duplicate registration of flags.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	BindLoggingFlags(&o.Config, fs)
}

// apply set klog logger from LogFormat type
func (o *Options) apply() {
	// if log format not exists, use nil loggr
	factory, _ := registry.LogRegistry.Get(o.Config.Format)
	if factory == nil {
		klog.ClearLogger()
	} else {
		log, flush := factory.Create(o.Config.Options)
		klog.SetLogger(log)
		logrFlush = flush
	}
	if o.Config.Sanitization {
		klog.SetLogFilter(&sanitization.SanitizingFilter{})
	}
	if err := loggingFlags.Lookup("v").Value.Set(o.Config.Verbosity.String()); err != nil {
		panic(fmt.Errorf("internal error while setting klog verbosity: %v", err))
	}
	if err := loggingFlags.Lookup("vmodule").Value.Set(o.Config.VModule.String()); err != nil {
		panic(fmt.Errorf("internal error while setting klog vmodule: %v", err))
	}
	go wait.Forever(FlushLogs, o.Config.FlushFrequency)
}
