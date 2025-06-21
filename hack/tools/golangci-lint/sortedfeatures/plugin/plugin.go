/*
Copyright 2025 The Kubernetes Authors.

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

// This must be package main
package main

import (
	"bytes"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/go/analysis"

	"k8s.io/kubernetes/hack/tools/golangci-lint/sortedfeatures"
)

// settings defines the configuration options for the sortedfeatures linter
type settings struct {
	// Debug enables debug logging
	Debug bool `json:"debug"`
	// Files specifies which files to check
	Files []string `json:"files"`
}

// New is the entry point for golangci-lint plugin system
func New(pluginSettings interface{}) ([]*analysis.Analyzer, error) {
	// Create default config
	config := sortedfeatures.Config{}

	// Parse settings if provided
	if pluginSettings != nil {
		var s settings
		// Convert settings to JSON and back to our struct for easier handling
		var buffer bytes.Buffer
		if err := json.NewEncoder(&buffer).Encode(pluginSettings); err != nil {
			return nil, fmt.Errorf("encoding settings as internal JSON buffer: %v", err)
		}

		decoder := json.NewDecoder(&buffer)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&s); err != nil {
			return nil, fmt.Errorf("decoding settings from internal JSON buffer: %v", err)
		}

		// Apply settings to config
		config.Debug = s.Debug

		// Use Files if specified
		if len(s.Files) > 0 {
			config.Files = append(config.Files, s.Files...)
		}

		if config.Debug {
			fmt.Printf("sortedfeatures settings: %+v\n", s)
			fmt.Printf("final config: %+v\n", config)
		}
	}

	// Get the analyzer with config
	analyzer := sortedfeatures.NewAnalyzerWithConfig(config)

	// Return the analyzer
	return []*analysis.Analyzer{analyzer}, nil
}
