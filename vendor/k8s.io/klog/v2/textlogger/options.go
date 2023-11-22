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

package textlogger

import (
	"flag"
	"io"
	"os"
	"strconv"
	"time"

	"k8s.io/klog/v2/internal/verbosity"
)

// Config influences logging in a text logger. To make this configurable via
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
	verbosityFlagName string
	vmoduleFlagName   string
	verbosityDefault  int
	fixedTime         *time.Time
	output            io.Writer
}

// VerbosityFlagName overrides the default -v for the verbosity level.
func VerbosityFlagName(name string) ConfigOption {
	return func(co *configOptions) {

		co.verbosityFlagName = name
	}
}

// VModulFlagName overrides the default -vmodule for the per-module
// verbosity levels.
func VModuleFlagName(name string) ConfigOption {
	return func(co *configOptions) {
		co.vmoduleFlagName = name
	}
}

// Verbosity overrides the default verbosity level of 0.
// See https://github.com/kubernetes/community/blob/9406b4352fe2d5810cb21cc3cb059ce5886de157/contributors/devel/sig-instrumentation/logging.md#logging-conventions
// for log level conventions in Kubernetes.
func Verbosity(level int) ConfigOption {
	return func(co *configOptions) {
		co.verbosityDefault = level
	}
}

// Output overrides stderr as the output stream.
func Output(output io.Writer) ConfigOption {
	return func(co *configOptions) {
		co.output = output
	}
}

// FixedTime overrides the actual time with a fixed time. Useful only for testing.
//
// # Experimental
//
// Notice: This function is EXPERIMENTAL and may be changed or removed in a
// later release.
func FixedTime(ts time.Time) ConfigOption {
	return func(co *configOptions) {
		co.fixedTime = &ts
	}
}

// NewConfig returns a configuration with recommended defaults and optional
// modifications. Command line flags are not bound to any FlagSet yet.
func NewConfig(opts ...ConfigOption) *Config {
	c := &Config{
		vstate: verbosity.New(),
		co: configOptions{
			verbosityFlagName: "v",
			vmoduleFlagName:   "vmodule",
			verbosityDefault:  0,
			output:            os.Stderr,
		},
	}
	for _, opt := range opts {
		opt(&c.co)
	}

	// Cannot fail for this input.
	_ = c.Verbosity().Set(strconv.FormatInt(int64(c.co.verbosityDefault), 10))
	return c
}

// AddFlags registers the command line flags that control the configuration.
func (c *Config) AddFlags(fs *flag.FlagSet) {
	fs.Var(c.Verbosity(), c.co.verbosityFlagName, "number for the log level verbosity of the testing logger")
	fs.Var(c.VModule(), c.co.vmoduleFlagName, "comma-separated list of pattern=N log level settings for files matching the patterns")
}
