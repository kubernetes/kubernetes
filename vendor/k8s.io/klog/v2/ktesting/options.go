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

package ktesting

import (
	"flag"
	"strconv"

	"k8s.io/klog/v2/internal/serialize"
	"k8s.io/klog/v2/internal/verbosity"
)

// Config influences logging in a test logger. To make this configurable via
// command line flags, instantiate this once per program and use AddFlags to
// bind command line flags to the instance before passing it to NewTestContext.
//
// Must be constructed with NewConfig.
type Config struct {
	vstate *verbosity.VState
	co     configOptions
}

// Verbosity returns a value instance that can be used to query (via String) or
// modify (via Set) the verbosity threshold. This is thread-safe and can be
// done at runtime.
func (c *Config) Verbosity() flag.Value {
	return c.vstate.V()
}

// VModule returns a value instance that can be used to query (via String) or
// modify (via Set) the vmodule settings. This is thread-safe and can be done
// at runtime.
func (c *Config) VModule() flag.Value {
	return c.vstate.VModule()
}

// ConfigOption implements functional parameters for NewConfig.
type ConfigOption func(co *configOptions)

type configOptions struct {
	anyToString       serialize.AnyToStringFunc
	verbosityFlagName string
	vmoduleFlagName   string
	verbosityDefault  int
	bufferLogs        bool
}

// AnyToString overrides the default formatter for values that are not
// supported directly by klog. The default is `fmt.Sprintf("%+v")`.
// The formatter must not panic.
func AnyToString(anyToString func(value interface{}) string) ConfigOption {
	return func(co *configOptions) {
		co.anyToString = anyToString
	}
}

// VerbosityFlagName overrides the default -testing.v for the verbosity level.
func VerbosityFlagName(name string) ConfigOption {
	return func(co *configOptions) {
		co.verbosityFlagName = name
	}
}

// VModulFlagName overrides the default -testing.vmodule for the per-module
// verbosity levels.
func VModuleFlagName(name string) ConfigOption {
	return func(co *configOptions) {
		co.vmoduleFlagName = name
	}
}

// Verbosity overrides the default verbosity level of 5. That default is higher
// than in klog itself because it enables logging entries for "the steps
// leading up to errors and warnings" and "troubleshooting" (see
// https://github.com/kubernetes/community/blob/9406b4352fe2d5810cb21cc3cb059ce5886de157/contributors/devel/sig-instrumentation/logging.md#logging-conventions),
// which is useful when debugging a failed test. `go test` only shows the log
// output for failed tests. To see all output, use `go test -v`.
func Verbosity(level int) ConfigOption {
	return func(co *configOptions) {
		co.verbosityDefault = level
	}
}

// BufferLogs controls whether log entries are captured in memory in addition
// to being printed. Off by default. Unit tests that want to verify that
// log entries are emitted as expected can turn this on and then retrieve
// the captured log through the Underlier LogSink interface.
func BufferLogs(enabled bool) ConfigOption {
	return func(co *configOptions) {
		co.bufferLogs = enabled
	}
}

// NewConfig returns a configuration with recommended defaults and optional
// modifications. Command line flags are not bound to any FlagSet yet.
func NewConfig(opts ...ConfigOption) *Config {
	c := &Config{
		co: configOptions{
			verbosityFlagName: "testing.v",
			vmoduleFlagName:   "testing.vmodule",
			verbosityDefault:  5,
		},
	}
	for _, opt := range opts {
		opt(&c.co)
	}

	c.vstate = verbosity.New()
	// Cannot fail for this input.
	_ = c.vstate.V().Set(strconv.FormatInt(int64(c.co.verbosityDefault), 10))
	return c
}

// AddFlags registers the command line flags that control the configuration.
func (c *Config) AddFlags(fs *flag.FlagSet) {
	fs.Var(c.vstate.V(), c.co.verbosityFlagName, "number for the log level verbosity of the testing logger")
	fs.Var(c.vstate.VModule(), c.co.vmoduleFlagName, "comma-separated list of pattern=N log level settings for files matching the patterns")
}
