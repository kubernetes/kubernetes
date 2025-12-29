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

package restartallcontainers

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// Ensure the feature struct implements the unified Feature interface.
var _ nodedeclaredfeatures.Feature = &restartAllContainersFeature{}

const (
	RestartRulesFeatureGate              = "ContainerRestartRules"
	RestartAllContainersOnContainerExits = "RestartAllContainersOnContainerExits"
)

// Feature is the implementation of the `GuaranteedQoSPodCPUResize` feature.
var Feature = &restartAllContainersFeature{}

type restartAllContainersFeature struct{}

func (f *restartAllContainersFeature) Name() string {
	return RestartAllContainersOnContainerExits
}

func (f *restartAllContainersFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(RestartAllContainersOnContainerExits)
}

func (f *restartAllContainersFeature) InferForScheduling(podInfo *nodedeclaredfeatures.PodInfo) bool {
	for _, c := range podInfo.Spec.Containers {
		for _, rule := range c.RestartPolicyRules {
			if rule.Action == v1.ContainerRestartRuleActionRestartAllContainers {
				return true
			}
		}
	}
	for _, c := range podInfo.Spec.InitContainers {
		for _, rule := range c.RestartPolicyRules {
			if rule.Action == v1.ContainerRestartRuleActionRestartAllContainers {
				return true
			}
		}
	}
	return false
}

func (f *restartAllContainersFeature) InferForUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	// container.restartPolicy and container.restartPolicyRules are not mutable.
	return false
}

func (f *restartAllContainersFeature) MaxVersion() *version.Version {
	return nil
}
