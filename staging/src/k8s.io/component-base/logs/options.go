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
	"flag"
	"fmt"
	"strings"

	"github.com/go-logr/logr"
	"github.com/spf13/pflag"

	"k8s.io/klog/v2"
)

const (
	logFormatFlagName = "logging-format"
	defaultLogFormat  = "text"
)

// List of logs (k8s.io/klog + k8s.io/component-base/logs) flags supported by all logging formats
var supportedLogsFlags = map[string]struct{}{
	"v": {},
	// TODO: support vmodule after 1.19 Alpha
}

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

// Validate verifies if any unsupported flag is set
// for non-default logging format
func (o *Options) Validate() []error {
	errs := []error{}
	if o.LogFormat != defaultLogFormat {
		allFlags := unsupportedLoggingFlags()
		for _, fname := range allFlags {
			if flagIsSet(fname) {
				errs = append(errs, fmt.Errorf("non-default logging format doesn't honor flag: %s", fname))
			}
		}
	}
	if _, err := o.Get(); err != nil {
		errs = append(errs, fmt.Errorf("unsupported log format: %s", o.LogFormat))
	}
	return errs
}

func flagIsSet(name string) bool {
	f := flag.Lookup(name)
	if f != nil {
		return f.DefValue != f.Value.String()
	}
	pf := pflag.Lookup(name)
	if pf != nil {
		return pf.DefValue != pf.Value.String()
	}
	panic("failed to lookup unsupported log flag")
}

// AddFlags add logging-format flag
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	unsupportedFlags := fmt.Sprintf("--%s", strings.Join(unsupportedLoggingFlags(), ", --"))
	formats := fmt.Sprintf(`"%s"`, strings.Join(logRegistry.List(), `", "`))
	fs.StringVar(&o.LogFormat, logFormatFlagName, defaultLogFormat, fmt.Sprintf("Sets the log format. Permitted formats: %s.\nNon-default formats don't honor these flags: %s.\nNon-default choices are currently alpha and subject to change without warning.", formats, unsupportedFlags))

	// No new log formats should be added after generation is of flag options
	logRegistry.Freeze()
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

func unsupportedLoggingFlags() []string {
	allFlags := []string{}

	// k8s.io/klog flags
	fs := &flag.FlagSet{}
	klog.InitFlags(fs)
	fs.VisitAll(func(flag *flag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			allFlags = append(allFlags, flag.Name)
		}
	})

	// k8s.io/component-base/logs flags
	pfs := &pflag.FlagSet{}
	AddFlags(pfs)
	pfs.VisitAll(func(flag *pflag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			allFlags = append(allFlags, flag.Name)
		}
	})
	return allFlags
}
