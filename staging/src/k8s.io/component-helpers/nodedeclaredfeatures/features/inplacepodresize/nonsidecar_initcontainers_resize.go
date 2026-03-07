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

package inplacepodresize

import (
	"reflect"

	core "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// Ensure the feature struct implements the unified Feature interface.
var _ nodedeclaredfeatures.Feature = &nonSidecarInitContainerResizeFeature{}

const NonSidecarInitContainerResizeFeatureName = "InPlacePodResizeNonSidecarInitContainers"

// Feature is the implementation of the `NonSidecarInitContainerResize` feature.
var NonSidecarInitContainerResizeFeature = &nonSidecarInitContainerResizeFeature{}

type nonSidecarInitContainerResizeFeature struct{}

func (f *nonSidecarInitContainerResizeFeature) Name() string {
	return NonSidecarInitContainerResizeFeatureName
}

func (f *nonSidecarInitContainerResizeFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) bool {
	return true
}

func (f *nonSidecarInitContainerResizeFeature) InferForScheduling(podInfo *nodedeclaredfeatures.PodInfo) bool {
	// This feature is only relevant for pod updates.
	return false
}

func (f *nonSidecarInitContainerResizeFeature) InferForUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	for i, ctr := range oldPodInfo.Spec.InitContainers {
		if !isSidecar(ctr) && !reflect.DeepEqual(ctr.Resources, newPodInfo.Spec.InitContainers[i].Resources) {
			return true
		}
	}
	return false
}

func (f *nonSidecarInitContainerResizeFeature) MaxVersion() *version.Version {
	return nil
}

func isSidecar(initContainer core.Container) bool {
	return initContainer.RestartPolicy != nil && *initContainer.RestartPolicy == core.ContainerRestartPolicyAlways
}
