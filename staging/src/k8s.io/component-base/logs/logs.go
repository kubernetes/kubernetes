/*
Copyright 2014 The Kubernetes Authors.

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

// Package logs contains support for logging options, flags and setup.
// Commands must explicitly enable command line flags. They no longer
// get added automatically when importing this package.
package logs

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/spf13/pflag"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/klog/v2"
)

const deprecated = "will be removed in a future release, see https://github.com/kubernetes/enhancements/tree/master/keps/sig-instrumentation/2845-deprecate-klog-specific-flags-in-k8s-components"

// TODO (https://github.com/kubernetes/kubernetes/issues/105310): once klog
// flags are removed, stop warning about "Non-default formats don't honor these
// flags" in config.go and instead add this remark here.
//
// const vmoduleUsage = " (only works for the default text log format)"

var (
	packageFlags = flag.NewFlagSet("logging", flag.ContinueOnError)

	// Periodic flushing gets configured either via the global flag
	// in this file or via LoggingConfiguration.
	logFlushFreq time.Duration
)

func init() {
	klog.InitFlags(packageFlags)
	packageFlags.DurationVar(&logFlushFreq, logsapi.LogFlushFreqFlagName, logsapi.LogFlushFreqDefault, "Maximum number of seconds between log flushes")
}

type addFlagsOptions struct {
	skipLoggingConfigurationFlags bool
}

type Option func(*addFlagsOptions)

// SkipLoggingConfigurationFlags must be used as option for AddFlags when
// the program also uses a LoggingConfiguration struct for configuring
// logging. Then only flags not covered by that get added.
func SkipLoggingConfigurationFlags() Option {
	return func(o *addFlagsOptions) {
		o.skipLoggingConfigurationFlags = true
	}
}

// Options is an alias for LoggingConfiguration to comply with component-base
// conventions.
type Options = logsapi.LoggingConfiguration

// NewOptions is an alias for NewLoggingConfiguration.
var NewOptions = logsapi.NewLoggingConfiguration

// AddFlags registers this package's flags on arbitrary FlagSets. This includes
// the klog flags, with the original underscore as separator between. If
// commands want hyphens as separators, they can set
// k8s.io/component-base/cli/flag/WordSepNormalizeFunc as normalization
// function on the flag set before calling AddFlags.
//
// May be called more than once.
func AddFlags(fs *pflag.FlagSet, opts ...Option) {
	// Determine whether the flags are already present by looking up one
	// which always should exist.
	if fs.Lookup("logtostderr") != nil {
		return
	}

	o := addFlagsOptions{}
	for _, opt := range opts {
		opt(&o)
	}

	// Add flags with pflag deprecation remark for some klog flags.
	packageFlags.VisitAll(func(f *flag.Flag) {
		pf := pflag.PFlagFromGoFlag(f)
		switch f.Name {
		case "v":
			// unchanged, potentially skip it
			if o.skipLoggingConfigurationFlags {
				return
			}
		case logsapi.LogFlushFreqFlagName:
			// unchanged, potentially skip it
			if o.skipLoggingConfigurationFlags {
				return
			}
		case "vmodule":
			// TODO: see above
			// pf.Usage += vmoduleUsage
			if o.skipLoggingConfigurationFlags {
				return
			}
		default:
			// deprecated, but not hidden
			pf.Deprecated = deprecated
		}
		fs.AddFlag(pf)
	})
}

// AddGoFlags is a variant of AddFlags for traditional Go flag.FlagSet.
// Commands should use pflag whenever possible for the sake of consistency.
// Cases where this function is needed include tests (they have to set up flags
// in flag.CommandLine) and commands that for historic reasons use Go
// flag.Parse and cannot change to pflag because it would break their command
// line interface.
func AddGoFlags(fs *flag.FlagSet, opts ...Option) {
	o := addFlagsOptions{}
	for _, opt := range opts {
		opt(&o)
	}

	// Add flags with deprecation remark added to the usage text of
	// some klog flags.
	packageFlags.VisitAll(func(f *flag.Flag) {
		usage := f.Usage
		switch f.Name {
		case "v":
			// unchanged
			if o.skipLoggingConfigurationFlags {
				return
			}
		case logsapi.LogFlushFreqFlagName:
			// unchanged
			if o.skipLoggingConfigurationFlags {
				return
			}
		case "vmodule":
			// TODO: see above
			// usage += vmoduleUsage
			if o.skipLoggingConfigurationFlags {
				return
			}
		default:
			usage += " (DEPRECATED: " + deprecated + ")"
		}
		fs.Var(f.Value, f.Name, usage)
	})
}

// KlogWriter serves as a bridge between the standard log package and the glog package.
type KlogWriter struct{}

// Write implements the io.Writer interface.
func (writer KlogWriter) Write(data []byte) (n int, err error) {
	klog.InfoDepth(1, string(data))
	return len(data), nil
}

// InitLogs initializes logs the way we want for Kubernetes.
// It should be called after parsing flags. If called before that,
// it will use the default log settings.
//
// InitLogs disables support for contextual logging in klog while
// that Kubernetes feature is not considered stable yet. Commands
// which want to support contextual logging can:
//   - call klog.EnableContextualLogging after calling InitLogs,
//     with a fixed `true` or depending on some command line flag or
//     a feature gate check
//   - set up a FeatureGate instance, the advanced logging configuration
//     with Options and call Options.ValidateAndApply with the FeatureGate;
//     k8s.io/component-base/logs/example/cmd demonstrates how to do that
func InitLogs() {
	log.SetOutput(KlogWriter{})
	log.SetFlags(0)

	// Start flushing now. If LoggingConfiguration.ApplyAndValidate is
	// used, it will restart the daemon with the log flush interval defined
	// there.
	klog.StartFlushDaemon(logFlushFreq)

	// This is the default in Kubernetes. Options.ValidateAndApply
	// will override this with the result of a feature gate check.
	klog.EnableContextualLogging(false)
}

// FlushLogs flushes logs immediately. This should be called at the end of
// the main function via defer to ensure that all pending log messages
// are printed before exiting the program.
func FlushLogs() {
	klog.Flush()
}

// NewLogger creates a new log.Logger which sends logs to klog.Info.
func NewLogger(prefix string) *log.Logger {
	return log.New(KlogWriter{}, prefix, 0)
}

// GlogSetter is a setter to set glog level.
func GlogSetter(val string) (string, error) {
	var level klog.Level
	if err := level.Set(val); err != nil {
		return "", fmt.Errorf("failed set klog.logging.verbosity %s: %v", val, err)
	}
	return fmt.Sprintf("successfully set klog.logging.verbosity to %s", val), nil
}
