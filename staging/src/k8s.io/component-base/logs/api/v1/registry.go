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

package v1

import (
	"fmt"
	"sort"
	"strings"

	"github.com/go-logr/logr"

	"k8s.io/component-base/featuregate"
)

var logRegistry = newLogFormatRegistry()

// logFormatRegistry stores factories for all supported logging formats.
type logFormatRegistry struct {
	registry map[string]logFormat
	frozen   bool
}

type logFormat struct {
	factory LogFormatFactory
	feature featuregate.Feature
}

// LogFormatFactory provides support for a certain additional,
// non-default log format.
type LogFormatFactory interface {
	// Create returns a logger with the requested configuration.
	// Returning a flush function for the logger is optional.
	// If provided, the caller must ensure that it is called
	// periodically (if desired) and at program exit.
	Create(c LoggingConfiguration) (log logr.Logger, flush func())
}

// RegisterLogFormat registers support for a new logging format. This must be called
// before using any of the methods in LoggingConfiguration. The feature must
// be one of those defined in this package (typically LoggingAlphaOptions,
// LoggingBetaOptions or LoggingStableOptions).
func RegisterLogFormat(name string, factory LogFormatFactory, feature featuregate.Feature) error {
	return logRegistry.register(name, logFormat{factory, feature})
}

func newLogFormatRegistry() *logFormatRegistry {
	registry := &logFormatRegistry{
		registry: make(map[string]logFormat),
		frozen:   false,
	}
	registry.register("text", logFormat{feature: LoggingStableOptions})
	return registry
}

// register adds a new log format. It's an error to modify an existing one.
func (lfr *logFormatRegistry) register(name string, format logFormat) error {
	if lfr.frozen {
		return fmt.Errorf("log format registry is frozen, unable to register log format %s", name)
	}
	if _, ok := lfr.registry[name]; ok {
		return fmt.Errorf("log format: %s already exists", name)
	}
	if _, ok := featureGates()[format.feature]; !ok && format.feature != LoggingStableOptions {
		return fmt.Errorf("log format %s: unsupported feature gate %s", name, format.feature)
	}
	lfr.registry[name] = format
	return nil
}

// get specified log format factory
func (lfr *logFormatRegistry) get(name string) (*logFormat, error) {
	format, ok := lfr.registry[name]
	if !ok {
		return nil, fmt.Errorf("log format: %s does not exists", name)
	}
	return &format, nil
}

// list names of registered log formats, including feature gates (sorted)
func (lfr *logFormatRegistry) list() string {
	formats := make([]string, 0, len(lfr.registry))
	for name, format := range lfr.registry {
		item := fmt.Sprintf(`"%s"`, name)
		if format.feature != LoggingStableOptions {
			item += fmt.Sprintf(" (gated by %s)", format.feature)
		}
		formats = append(formats, item)
	}
	sort.Strings(formats)
	return strings.Join(formats, ", ")
}

// freeze prevents further modifications of the registered log formats.
func (lfr *logFormatRegistry) freeze() {
	lfr.frozen = true
}
