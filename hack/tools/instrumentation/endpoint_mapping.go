/*
Copyright The Kubernetes Authors.

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

package main

import (
	"os"
	"slices"
	"sort"
	"strings"

	yaml "go.yaml.in/yaml/v2"

	"k8s.io/kubernetes/hack/tools/instrumentation/internal/metric"
)

type endpointMappingConfig struct {
	CoreComponents       map[string][]string `yaml:"coreComponents"`
	StandaloneComponents map[string][]string `yaml:"standaloneComponents"`
	SharedPaths          []string            `yaml:"sharedPaths"`
	EndpointMappings     []endpointMapping   `yaml:"endpointMappings"`
	DefaultEndpoint      string              `yaml:"defaultEndpoint"`
}

type endpointMapping struct {
	PathContains string `yaml:"pathContains"`
	Endpoint     string `yaml:"endpoint"`
}

func loadEndpointMappingConfig(path string) (*endpointMappingConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var config endpointMappingConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	if config.DefaultEndpoint == "" {
		config.DefaultEndpoint = "/metrics"
	}
	return &config, nil
}

// inferComponentEndpoints determines the component(s) and endpoint for a metric based on its source file path.
func (c *endpointMappingConfig) inferComponentEndpoints(filePath string) []metric.ComponentEndpoint {
	endpoint := c.inferEndpoint(filePath)

	if c.isSharedPath(filePath) {
		// The assumption here is that none of the standalone components
		// use the metrics under the path.
		return c.allCoreComponentEndpoints(endpoint)
	}

	// Core and standalone components may explicitly share the same metrics through their path patterns.
	components := c.inferComponents(filePath, c.CoreComponents)
	components = append(components, c.inferComponents(filePath, c.StandaloneComponents)...)

	var endpoints []metric.ComponentEndpoint
	for _, component := range components {
		endpoints = append(endpoints, metric.ComponentEndpoint{
			Component: component,
			Endpoint:  endpoint,
		})
	}

	return endpoints
}

func (c *endpointMappingConfig) isSharedPath(filePath string) bool {
	for _, pattern := range c.SharedPaths {
		if strings.HasPrefix(filePath, pattern) {
			return true
		}
	}
	return false
}

func (c *endpointMappingConfig) inferComponents(filePath string, components map[string][]string) []string {
	var matchingComponents []string
	for component, patterns := range components {
		for _, pattern := range patterns {
			if strings.HasPrefix(filePath, pattern) {
				matchingComponents = append(matchingComponents, component)
			}
		}
	}
	// Sort to ensure consistent result, regardless of map iteration order.
	slices.Sort(matchingComponents)
	return matchingComponents
}

func (c *endpointMappingConfig) inferEndpoint(filePath string) string {
	for _, mapping := range c.EndpointMappings {
		if strings.Contains(filePath, mapping.PathContains) {
			return mapping.Endpoint
		}
	}
	return c.DefaultEndpoint
}

// allCoreComponentEndpoints returns a ComponentEndpoint for each core component with the given endpoint.
func (c *endpointMappingConfig) allCoreComponentEndpoints(endpoint string) []metric.ComponentEndpoint {
	result := make([]metric.ComponentEndpoint, 0, len(c.CoreComponents))
	for component := range c.CoreComponents {
		result = append(result, metric.ComponentEndpoint{
			Component: component,
			Endpoint:  endpoint,
		})
	}
	// Sort for deterministic output
	sort.Slice(result, func(i, j int) bool {
		return result[i].Component < result[j].Component
	})
	return result
}
