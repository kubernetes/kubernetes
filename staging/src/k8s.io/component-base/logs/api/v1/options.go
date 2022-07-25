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

package v1

import (
	"flag"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/featuregate"
)

const (
	// LogFlushFreqDefault is the default for the corresponding command line
	// parameter.
	LogFlushFreqDefault = 5 * time.Second
)

const (
	// LogFlushFreqFlagName is the name of the command line parameter.
	// Depending on how flags get added, it is either a stand-alone
	// value (logs.AddFlags) or part of LoggingConfiguration.
	LogFlushFreqFlagName = "log-flush-frequency"
)

// NewLoggingConfiguration returns a struct holding the default logging configuration.
func NewLoggingConfiguration() *LoggingConfiguration {
	c := LoggingConfiguration{}
	SetRecommendedLoggingConfiguration(&c)
	return &c
}

// ValidateAndApply combines validation and application of the logging configuration.
// This should be invoked as early as possible because then the rest of the program
// startup (including validation of other options) will already run with the final
// logging configuration.
//
// The optional FeatureGate controls logging features. If nil, the default for
// these features is used.
func ValidateAndApply(c *LoggingConfiguration, featureGate featuregate.FeatureGate) error {
	return ValidateAndApplyAsField(c, featureGate, nil)
}

// ValidateAndApplyAsField is a variant of ValidateAndApply that should be used
// when the LoggingConfiguration is embedded in some larger configuration
// structure.
func ValidateAndApplyAsField(c *LoggingConfiguration, featureGate featuregate.FeatureGate, fldPath *field.Path) error {
	errs := Validate(c, featureGate, fldPath)
	if len(errs) > 0 {
		return errs.ToAggregate()
	}
	return apply(c, featureGate)
}

// Validate can be used to check for invalid settings without applying them.
// Most binaries should validate and apply the logging configuration as soon
// as possible via ValidateAndApply. The field path is optional: nil
// can be passed when the struct is not embedded in some larger struct.
func Validate(c *LoggingConfiguration, featureGate featuregate.FeatureGate, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if c.Format != DefaultLogFormat {
		// WordSepNormalizeFunc is just a guess. Commands should use it,
		// but we cannot know for sure.
		allFlags := unsupportedLoggingFlags(cliflag.WordSepNormalizeFunc)
		for _, f := range allFlags {
			if f.DefValue != f.Value.String() {
				errs = append(errs, field.Invalid(fldPath.Child("format"), c.Format, fmt.Sprintf("Non-default format doesn't honor flag: %s", f.Name)))
			}
		}
	}
	format, err := logRegistry.get(c.Format)
	if err != nil {
		errs = append(errs, field.Invalid(fldPath.Child("format"), c.Format, "Unsupported log format"))
	} else if format != nil {
		if format.feature != LoggingStableOptions {
			enabled := featureGates()[format.feature].Default
			if featureGate != nil {
				enabled = featureGate.Enabled(format.feature)
			}
			if !enabled {
				errs = append(errs, field.Forbidden(fldPath.Child("format"), fmt.Sprintf("Log format %s is disabled, see %s feature", c.Format, format.feature)))
			}
		}
	}

	// The type in our struct is uint32, but klog only accepts positive int32.
	if c.Verbosity > math.MaxInt32 {
		errs = append(errs, field.Invalid(fldPath.Child("verbosity"), c.Verbosity, fmt.Sprintf("Must be <= %d", math.MaxInt32)))
	}
	vmoduleFldPath := fldPath.Child("vmodule")
	if len(c.VModule) > 0 && c.Format != "" && c.Format != "text" {
		errs = append(errs, field.Forbidden(vmoduleFldPath, "Only supported for text log format"))
	}
	for i, item := range c.VModule {
		if item.FilePattern == "" {
			errs = append(errs, field.Required(vmoduleFldPath.Index(i), "File pattern must not be empty"))
		}
		if strings.ContainsAny(item.FilePattern, "=,") {
			errs = append(errs, field.Invalid(vmoduleFldPath.Index(i), item.FilePattern, "File pattern must not contain equal sign or comma"))
		}
		if item.Verbosity > math.MaxInt32 {
			errs = append(errs, field.Invalid(vmoduleFldPath.Index(i), item.Verbosity, fmt.Sprintf("Must be <= %d", math.MaxInt32)))
		}
	}

	errs = append(errs, validateFormatOptions(c, featureGate, fldPath.Child("options"))...)
	return errs
}

func validateFormatOptions(c *LoggingConfiguration, featureGate featuregate.FeatureGate, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	errs = append(errs, validateJSONOptions(c, featureGate, fldPath.Child("json"))...)
	return errs
}

func validateJSONOptions(c *LoggingConfiguration, featureGate featuregate.FeatureGate, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if gate := LoggingAlphaOptions; c.Options.JSON.SplitStream && !featureEnabled(featureGate, gate) {
		errs = append(errs, field.Forbidden(fldPath.Child("splitStream"), fmt.Sprintf("Feature %s is disabled", gate)))
	}
	if gate := LoggingAlphaOptions; c.Options.JSON.InfoBufferSize.Value() != 0 && !featureEnabled(featureGate, gate) {
		errs = append(errs, field.Forbidden(fldPath.Child("infoBufferSize"), fmt.Sprintf("Feature %s is disabled", gate)))
	}
	return errs
}

func featureEnabled(featureGate featuregate.FeatureGate, feature featuregate.Feature) bool {
	enabled := false
	if featureGate != nil {
		enabled = featureGate.Enabled(feature)
	}
	return enabled
}

func apply(c *LoggingConfiguration, featureGate featuregate.FeatureGate) error {
	contextualLoggingEnabled := contextualLoggingDefault
	if featureGate != nil {
		contextualLoggingEnabled = featureGate.Enabled(ContextualLogging)
	}

	// if log format not exists, use nil loggr
	format, _ := logRegistry.get(c.Format)
	if format.factory == nil {
		klog.ClearLogger()
	} else {
		log, flush := format.factory.Create(*c)
		klog.SetLoggerWithOptions(log, klog.ContextualLogger(contextualLoggingEnabled), klog.FlushLogger(flush))
	}
	if err := loggingFlags.Lookup("v").Value.Set(VerbosityLevelPflag(&c.Verbosity).String()); err != nil {
		return fmt.Errorf("internal error while setting klog verbosity: %v", err)
	}
	if err := loggingFlags.Lookup("vmodule").Value.Set(VModuleConfigurationPflag(&c.VModule).String()); err != nil {
		return fmt.Errorf("internal error while setting klog vmodule: %v", err)
	}
	klog.StartFlushDaemon(c.FlushFrequency)
	klog.EnableContextualLogging(contextualLoggingEnabled)
	return nil
}

// AddFlags adds command line flags for the configuration.
func AddFlags(c *LoggingConfiguration, fs *pflag.FlagSet) {
	// The help text is generated assuming that flags will eventually use
	// hyphens, even if currently no normalization function is set for the
	// flag set yet.
	unsupportedFlags := strings.Join(unsupportedLoggingFlagNames(cliflag.WordSepNormalizeFunc), ", ")
	formats := logRegistry.list()
	fs.StringVar(&c.Format, "logging-format", c.Format, fmt.Sprintf("Sets the log format. Permitted formats: %s.\nNon-default formats don't honor these flags: %s.\nNon-default choices are currently alpha and subject to change without warning.", formats, unsupportedFlags))
	// No new log formats should be added after generation is of flag options
	logRegistry.freeze()

	fs.DurationVar(&c.FlushFrequency, LogFlushFreqFlagName, c.FlushFrequency, "Maximum number of seconds between log flushes")
	fs.VarP(VerbosityLevelPflag(&c.Verbosity), "v", "v", "number for the log level verbosity")
	fs.Var(VModuleConfigurationPflag(&c.VModule), "vmodule", "comma-separated list of pattern=N settings for file-filtered logging (only works for text log format)")

	// JSON options. We only register them if "json" is a valid format. The
	// config file API however always has them.
	if _, err := logRegistry.get("json"); err == nil {
		fs.BoolVar(&c.Options.JSON.SplitStream, "log-json-split-stream", false, "[Alpha] In JSON format, write error messages to stderr and info messages to stdout. The default is to write a single stream to stdout. Enable the LoggingAlphaOptions feature gate to use this.")
		fs.Var(&c.Options.JSON.InfoBufferSize, "log-json-info-buffer-size", "[Alpha] In JSON format with split output streams, the info messages can be buffered for a while to increase performance. The default value of zero bytes disables buffering. The size can be specified as number of bytes (512), multiples of 1000 (1K), multiples of 1024 (2Ki), or powers of those (3M, 4G, 5Mi, 6Gi). Enable the LoggingAlphaOptions feature gate to use this.")
	}
}

// SetRecommendedLoggingConfiguration sets the default logging configuration
// for fields that are unset.
//
// Consumers who embed LoggingConfiguration in their own configuration structs
// may set custom defaults and then should call this function to add the
// global defaults.
func SetRecommendedLoggingConfiguration(c *LoggingConfiguration) {
	if c.Format == "" {
		c.Format = "text"
	}
	if c.FlushFrequency == 0 {
		c.FlushFrequency = LogFlushFreqDefault
	}
	var empty resource.QuantityValue
	if c.Options.JSON.InfoBufferSize == empty {
		c.Options.JSON.InfoBufferSize = resource.QuantityValue{
			// This is similar, but not quite the same as a default
			// constructed instance.
			Quantity: *resource.NewQuantity(0, resource.DecimalSI),
		}
		// This sets the unexported Quantity.s which will be compared
		// by reflect.DeepEqual in some tests.
		_ = c.Options.JSON.InfoBufferSize.String()
	}
}

// loggingFlags captures the state of the logging flags, in particular their default value
// before flag parsing. It is used by unsupportedLoggingFlags.
var loggingFlags pflag.FlagSet

func init() {
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	loggingFlags.AddGoFlagSet(&fs)
}

// List of logs (k8s.io/klog + k8s.io/component-base/logs) flags supported by all logging formats
var supportedLogsFlags = map[string]struct{}{
	"v": {},
	// TODO: support vmodule after 1.19 Alpha
}

// unsupportedLoggingFlags lists unsupported logging flags. The normalize
// function is optional.
func unsupportedLoggingFlags(normalizeFunc func(f *pflag.FlagSet, name string) pflag.NormalizedName) []*pflag.Flag {
	// k8s.io/component-base/logs and klog flags
	pfs := &pflag.FlagSet{}
	loggingFlags.VisitAll(func(flag *pflag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			// Normalization changes flag.Name, so make a copy.
			clone := *flag
			pfs.AddFlag(&clone)
		}
	})

	// Apply normalization.
	pfs.SetNormalizeFunc(normalizeFunc)

	var allFlags []*pflag.Flag
	pfs.VisitAll(func(flag *pflag.Flag) {
		allFlags = append(allFlags, flag)
	})
	return allFlags
}

// unsupportedLoggingFlagNames lists unsupported logging flags by name, with
// optional normalization and sorted.
func unsupportedLoggingFlagNames(normalizeFunc func(f *pflag.FlagSet, name string) pflag.NormalizedName) []string {
	unsupportedFlags := unsupportedLoggingFlags(normalizeFunc)
	names := make([]string, 0, len(unsupportedFlags))
	for _, f := range unsupportedFlags {
		names = append(names, "--"+f.Name)
	}
	sort.Strings(names)
	return names
}
