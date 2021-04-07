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

	"k8s.io/component-base/logs/sanitization"
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
	LogFormat       string
	LogSanitization bool
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
		allFlags := unsupportedLoggingFlags(hyphensToUnderscores)
		for _, fname := range allFlags {
			if flagIsSet(fname, hyphensToUnderscores) {
				errs = append(errs, fmt.Errorf("non-default logging format doesn't honor flag: %s", fname))
			}
		}
	}
	if _, err := o.Get(); err != nil {
		errs = append(errs, fmt.Errorf("unsupported log format: %s", o.LogFormat))
	}
	return errs
}

// hyphensToUnderscores replaces hyphens with underscores
// we should always use underscores instead of hyphens when validate flags
func hyphensToUnderscores(s string) string {
	return strings.Replace(s, "-", "_", -1)
}

func flagIsSet(name string, normalizeFunc func(name string) string) bool {
	f := flag.Lookup(name)
	if f != nil {
		return f.DefValue != f.Value.String()
	}
	if normalizeFunc != nil {
		f = flag.Lookup(normalizeFunc(name))
		if f != nil {
			return f.DefValue != f.Value.String()
		}
	}
	pf := pflag.Lookup(name)
	if pf != nil {
		return pf.DefValue != pf.Value.String()
	}
	panic("failed to lookup unsupported log flag")
}

// AddFlags add logging-format flag
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	normalizeFunc := func(name string) string {
		f := fs.GetNormalizeFunc()
		return string(f(fs, name))
	}

	unsupportedFlags := fmt.Sprintf("--%s", strings.Join(unsupportedLoggingFlags(normalizeFunc), ", --"))
	formats := fmt.Sprintf(`"%s"`, strings.Join(logRegistry.List(), `", "`))
	fs.StringVar(&o.LogFormat, logFormatFlagName, defaultLogFormat, fmt.Sprintf("Sets the log format. Permitted formats: %s.\nNon-default formats don't honor these flags: %s.\nNon-default choices are currently alpha and subject to change without warning.", formats, unsupportedFlags))

	// No new log formats should be added after generation is of flag options
	logRegistry.Freeze()
	fs.BoolVar(&o.LogSanitization, "experimental-logging-sanitization", o.LogSanitization, `[Experimental] When enabled prevents logging of fields tagged as sensitive (passwords, keys, tokens).
Runtime log sanitization may introduce significant computation overhead and therefore should not be enabled in production.`)
}

// Apply set klog logger from LogFormat type
func (o *Options) Apply() {
	// if log format not exists, use nil loggr
	loggr, _ := o.Get()
	klog.SetLogger(loggr)
	if o.LogSanitization {
		klog.SetLogFilter(&sanitization.SanitizingFilter{})
	}
}

// Get logger with LogFormat field
func (o *Options) Get() (logr.Logger, error) {
	return logRegistry.Get(o.LogFormat)
}

func unsupportedLoggingFlags(normalizeFunc func(name string) string) []string {
	allFlags := []string{}

	// k8s.io/klog flags
	fs := &flag.FlagSet{}
	klog.InitFlags(fs)
	fs.VisitAll(func(flag *flag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			name := flag.Name
			if normalizeFunc != nil {
				name = normalizeFunc(name)
			}
			allFlags = append(allFlags, name)
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
