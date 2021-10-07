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

	"github.com/go-logr/logr"
)

var logRegistry = newLogFormatRegistry()

// logFormatRegistry stores factories for all supported logging formats.
type logFormatRegistry struct {
	registry map[string]LogFormatFactory
	frozen   bool
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
// before using any of the methods in LoggingConfiguration.
func RegisterLogFormat(name string, factory LogFormatFactory) error {
	return logRegistry.register(name, factory)
}

func newLogFormatRegistry() *logFormatRegistry {
	registry := &logFormatRegistry{
		registry: make(map[string]LogFormatFactory),
		frozen:   false,
	}
	registry.register("text", nil)
	return registry
}

// register adds a new log format. It's an error to modify an existing one.
func (lfr *logFormatRegistry) register(name string, factory LogFormatFactory) error {
	if lfr.frozen {
		return fmt.Errorf("log format registry is frozen, unable to register log format %s", name)
	}
	if _, ok := lfr.registry[name]; ok {
		return fmt.Errorf("log format: %s already exists", name)
	}
	lfr.registry[name] = factory
	return nil
}

// get specified log format factory
func (lfr *logFormatRegistry) get(name string) (LogFormatFactory, error) {
	re, ok := lfr.registry[name]
	if !ok {
		return nil, fmt.Errorf("log format: %s does not exists", name)
	}
	return re, nil
}

// list names of registered log formats (sorted)
func (lfr *logFormatRegistry) list() []string {
	formats := make([]string, 0, len(lfr.registry))
	for f := range lfr.registry {
		formats = append(formats, f)
	}
	sort.Strings(formats)
	return formats
}

// freeze prevents further modifications of the registered log formats.
func (lfr *logFormatRegistry) freeze() {
	lfr.frozen = true
}
