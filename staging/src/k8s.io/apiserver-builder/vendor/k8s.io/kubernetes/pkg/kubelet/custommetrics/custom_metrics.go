/*
Copyright 2015 The Kubernetes Authors.

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

// Package custommetrics contains support for instrumenting cAdvisor to gather custom metrics from pods.
package custommetrics

import (
	"path"

	"k8s.io/kubernetes/pkg/api/v1"
)

const (
	CustomMetricsDefinitionContainerFile = "definition.json"

	CustomMetricsDefinitionDir = "/etc/custom-metrics"
)

// Alpha implementation.
// Returns a path to a cAdvisor-specific custom metrics configuration.
func GetCAdvisorCustomMetricsDefinitionPath(container *v1.Container) (*string, error) {
	// Assuemes that the container has Custom Metrics enabled if it has "/etc/custom-metrics" directory
	// mounted as a volume. Custom Metrics definition is expected to be in "definition.json".
	if container.VolumeMounts != nil {
		for _, volumeMount := range container.VolumeMounts {
			if path.Clean(volumeMount.MountPath) == path.Clean(CustomMetricsDefinitionDir) {
				// TODO: add definition file validation.
				definitionPath := path.Clean(path.Join(volumeMount.MountPath, CustomMetricsDefinitionContainerFile))
				return &definitionPath, nil
			}
		}
	}
	// No Custom Metrics definition available.
	return nil, nil
}
