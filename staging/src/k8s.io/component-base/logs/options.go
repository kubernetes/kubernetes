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
	"github.com/go-logr/logr"
	"github.com/spf13/pflag"
	"k8s.io/klog/v2"
)

const (
	logFormatFlagName = "logging-format"
	defaultLogFormat  = "text"
)

// Options has klog format parameters
type Options struct {
	LogFormat string
}

// NewOptions return new klog options
func NewOptions() *Options {
	return &Options{
		LogFormat: defaultLogFormat,
	}
}

// Validate check LogFormat in registry or not
func (o *Options) Validate() []error {
	if _, err := o.Get(); err != nil {
		return []error{fmt.Errorf("unsupported log format: %s", o.LogFormat)}
	}
	return nil
}

// AddFlags add logging-format flag
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.LogFormat, logFormatFlagName, defaultLogFormat, "Set log format")
}

// Apply set klog logger from LogFormat type
func (o *Options) Apply() {
	// if log format not exists, use nil loggr
	loggr, _ := o.Get()

	klog.SetLogger(loggr)
}

// Get logger with LogFormat field
func (o *Options) Get() (logr.Logger, error) {
	return logRegistry.Get(o.LogFormat)
}
