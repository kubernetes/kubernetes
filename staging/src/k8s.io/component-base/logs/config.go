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

package logs

import (
	"flag"
	"fmt"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/component-base/config"
	"k8s.io/klog/v2"
)

// Supported klog formats
const (
	DefaultLogFormat = "text"
	JSONLogFormat    = "json"
)

// LogRegistry is new init LogFormatRegistry struct
var LogRegistry = NewLogFormatRegistry()

func init() {
	// Text format is default klog format
	LogRegistry.Register(DefaultLogFormat, nil)
}

// List of logs (k8s.io/klog + k8s.io/component-base/logs) flags supported by all logging formats
var supportedLogsFlags = map[string]struct{}{
	"v": {},
	// TODO: support vmodule after 1.19 Alpha
}

// BindLoggingFlags binds the Options struct fields to a flagset
func BindLoggingFlags(c *config.LoggingConfiguration, fs *pflag.FlagSet) {
	normalizeFunc := func(name string) string {
		f := fs.GetNormalizeFunc()
		return string(f(fs, name))
	}
	unsupportedFlags := fmt.Sprintf("--%s", strings.Join(UnsupportedLoggingFlags(normalizeFunc), ", --"))
	formats := fmt.Sprintf(`"%s"`, strings.Join(LogRegistry.List(), `", "`))
	fs.StringVar(&c.Format, "logging-format", c.Format, fmt.Sprintf("Sets the log format. Permitted formats: %s.\nNon-default formats don't honor these flags: %s.\nNon-default choices are currently alpha and subject to change without warning.", formats, unsupportedFlags))
	// No new log formats should be added after generation is of flag options
	LogRegistry.Freeze()
	fs.BoolVar(&c.Sanitization, "experimental-logging-sanitization", c.Sanitization, `[Experimental] When enabled prevents logging of fields tagged as sensitive (passwords, keys, tokens).
Runtime log sanitization may introduce significant computation overhead and therefore should not be enabled in production.`)
}

// UnsupportedLoggingFlags lists unsupported logging flags
func UnsupportedLoggingFlags(normalizeFunc func(name string) string) []string {
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
