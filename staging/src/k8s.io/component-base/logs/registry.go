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
	"fmt"
	"sync"

	"github.com/go-logr/logr"
)

var logRegistry = NewLogFormatRegistry()

// LogFormatRegistry store klog format registry
type LogFormatRegistry struct {
	registry map[string]logr.Logger
	mu       sync.Mutex
}

// NewLogFormatRegistry return new init LogFormatRegistry struct
func NewLogFormatRegistry() *LogFormatRegistry {
	return &LogFormatRegistry{
		registry: make(map[string]logr.Logger),
		mu:       sync.Mutex{},
	}
}

// Register new log format registry to global logRegistry
func (lfr *LogFormatRegistry) Register(name string, logger logr.Logger) error {
	lfr.mu.Lock()
	defer lfr.mu.Unlock()
	if _, ok := lfr.registry[name]; ok {
		return fmt.Errorf("log format: %s already exists", name)
	}
	lfr.registry[name] = logger
	return nil
}

// Get specified log format logger
func (lfr *LogFormatRegistry) Get(name string) (logr.Logger, error) {
	lfr.mu.Lock()
	defer lfr.mu.Unlock()
	re, ok := lfr.registry[name]
	if !ok {
		return nil, fmt.Errorf("log format: %s does not exists", name)
	}
	return re, nil
}

// Set specified log format logger
func (lfr *LogFormatRegistry) Set(name string, logger logr.Logger) {
	lfr.mu.Lock()
	defer lfr.mu.Unlock()
	lfr.registry[name] = logger
}

// Delete specified log format logger
func (lfr *LogFormatRegistry) Delete(name string) {
	lfr.mu.Lock()
	defer lfr.mu.Unlock()
	delete(lfr.registry, name)
}

func init() {
	// Text format is default klog format
	logRegistry.Register("text", nil)
}
