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
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
)

// Supported output formats.
const (
	// DefaultLogFormat is the traditional klog output format.
	DefaultLogFormat = "text"

	// JSONLogFormat emits each log message as a JSON struct.
	JSONLogFormat = "json"
)

// The alpha or beta level of structs is the highest stability level of any field
// inside it. Feature gates will get checked during LoggingConfiguration.ValidateAndApply.

// LoggingConfiguration contains logging options.
type LoggingConfiguration struct {
	// Format Flag specifies the structure of log messages.
	// default value of format is `text`
	Format string `json:"format,omitempty"`
	// Maximum number of nanoseconds (i.e. 1s = 1000000000) between log
	// flushes. Ignored if the selected logging backend writes log
	// messages without buffering.
	FlushFrequency time.Duration `json:"flushFrequency"`
	// Verbosity is the threshold that determines which log messages are
	// logged. Default is zero which logs only the most important
	// messages. Higher values enable additional messages. Error messages
	// are always logged.
	Verbosity VerbosityLevel `json:"verbosity"`
	// VModule overrides the verbosity threshold for individual files.
	// Only supported for "text" log format.
	VModule VModuleConfiguration `json:"vmodule,omitempty"`
	// [Alpha] Options holds additional parameters that are specific
	// to the different logging formats. Only the options for the selected
	// format get used, but all of them get validated.
	// Only available when the LoggingAlphaOptions feature gate is enabled.
	Options FormatOptions `json:"options,omitempty"`
}

// FormatOptions contains options for the different logging formats.
type FormatOptions struct {
	// [Alpha] JSON contains options for logging format "json".
	// Only available when the LoggingAlphaOptions feature gate is enabled.
	JSON JSONOptions `json:"json,omitempty"`
}

// JSONOptions contains options for logging format "json".
type JSONOptions struct {
	// [Alpha] SplitStream redirects error messages to stderr while
	// info messages go to stdout, with buffering. The default is to write
	// both to stdout, without buffering. Only available when
	// the LoggingAlphaOptions feature gate is enabled.
	SplitStream bool `json:"splitStream,omitempty"`
	// [Alpha] InfoBufferSize sets the size of the info stream when
	// using split streams. The default is zero, which disables buffering.
	// Only available when the LoggingAlphaOptions feature gate is enabled.
	InfoBufferSize resource.QuantityValue `json:"infoBufferSize,omitempty"`
}

// VModuleConfiguration is a collection of individual file names or patterns
// and the corresponding verbosity threshold.
type VModuleConfiguration []VModuleItem

// VModuleItem defines verbosity for one or more files which match a certain
// glob pattern.
type VModuleItem struct {
	// FilePattern is a base file name (i.e. minus the ".go" suffix and
	// directory) or a "glob" pattern for such a name. It must not contain
	// comma and equal signs because those are separators for the
	// corresponding klog command line argument.
	FilePattern string `json:"filePattern"`
	// Verbosity is the threshold for log messages emitted inside files
	// that match the pattern.
	Verbosity VerbosityLevel `json:"verbosity"`
}

// VerbosityLevel represents a klog or logr verbosity threshold.
type VerbosityLevel uint32
