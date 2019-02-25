/*
Copyright 2018 The Kubernetes Authors.

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

	componentbaseconfig "k8s.io/component-base/config"
)

// DebuggingOptions holds the Debugging options.
type DebuggingOptions struct {
	*componentbaseconfig.DebuggingConfiguration
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet.
func (o *DebuggingOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.EnableProfiling, "profiling", o.EnableProfiling,
		"Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&o.EnableContentionProfiling, "contention-profiling", o.EnableContentionProfiling,
		"Enable lock contention profiling, if profiling is enabled")
}

// ApplyTo fills up Debugging config with options.
func (o *DebuggingOptions) ApplyTo(cfg *componentbaseconfig.DebuggingConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.EnableProfiling = o.EnableProfiling
	cfg.EnableContentionProfiling = o.EnableContentionProfiling

	return nil
}

// Validate checks validation of DebuggingOptions.
func (o *DebuggingOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
