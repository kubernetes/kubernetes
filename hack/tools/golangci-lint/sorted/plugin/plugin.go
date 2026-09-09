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

// Package plugin registers the sorted linter as a golangci-lint module plugin.
package plugin

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/golangci/plugin-module-register/register"
	"golang.org/x/tools/go/analysis"

	"k8s.io/kubernetes/hack/tools/golangci-lint/sorted/pkg"
)

// settings defines the configuration options for the sorted linter
type settings struct {
	// Debug enables debug logging
	Debug bool `json:"debug"`
	// Files specifies which files to check
	Files []string `json:"files"`
}

// List of default files to check for feature gate sorting
var defaultTargetFiles = []string{
	"pkg/features/kube_features.go",
	"staging/src/k8s.io/apiserver/pkg/features/kube_features.go",
	"staging/src/k8s.io/client-go/features/known_features.go",
	"staging/src/k8s.io/controller-manager/pkg/features/kube_features.go",
	"staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go",
	"test/e2e/feature/feature.go",
	"test/e2e/environment/environment.go",
}

func New(pluginSettings interface{}) (register.LinterPlugin, error) {
	// Create default config
	config := pkg.Config{}

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
		config.Files = append(config.Files, s.Files...)
		if len(config.Files) == 0 {
			// If no files are specified, use the default target files
			config.Files = defaultTargetFiles
		}

		if config.Debug {
			fmt.Printf("sorted settings: %+v\n", s)
			fmt.Printf("final config: %+v\n", config)
		}
	} else {
		config.Files = defaultTargetFiles
	}

	return linterPlugin{config: config}, nil
}

type linterPlugin struct {
	config pkg.Config
}

func (l linterPlugin) BuildAnalyzers() ([]*analysis.Analyzer, error) {
	return []*analysis.Analyzer{pkg.NewAnalyzerWithConfig(l.config)}, nil
}
func (l linterPlugin) GetLoadMode() string {
	return register.LoadModeSyntax
}

func init() {
	register.Plugin("sorted", New)
}
