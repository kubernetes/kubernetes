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

package cgroupnamespaces

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &cgroupNamespacesFeature{}

const (
	// CgroupNamespacesSupportFeatureGate is the feature gate name.
	CgroupNamespacesSupportFeatureGate = "CgroupNamespacesSupport"
	// CgroupNamespacesSupport is the declared feature name.
	CgroupNamespacesSupport = "CgroupNamespacesSupport"
)

// Feature is the implementation of the `CgroupNamespacesSupport` feature.
var Feature = &cgroupNamespacesFeature{}

type cgroupNamespacesFeature struct{}

func (f *cgroupNamespacesFeature) Name() string {
	return CgroupNamespacesSupport
}

func (f *cgroupNamespacesFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{CgroupNamespacesSupportFeatureGate},
		RequiredRuntimeFeatures: &types.RuntimeFeatures{
			CgroupNamespaces: true,
		},
	}
}

func (f *cgroupNamespacesFeature) Discover(cfg *types.NodeConfiguration) bool {
	// A kubelet without the feature gate ignores hostCgroupNamespace, and so does
	// a runtime without cgroup namespace support.
	if !cfg.FeatureGates.Enabled(CgroupNamespacesSupportFeatureGate) {
		return false
	}

	return cfg.RuntimeFeatures.CgroupNamespaces
}

func (f *cgroupNamespacesFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	return podSpecUsesHostCgroupNamespace(podInfo.Spec)
}

func (f *cgroupNamespacesFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// hostCgroupNamespace cannot be updated, and cannot be set for ephemeral containers.
	return false
}

func (f *cgroupNamespacesFeature) MaxVersion() *version.Version {
	return nil
}

// podSpecUsesHostCgroupNamespace returns true if any container sets hostCgroupNamespace,
// to either true or false, as both need the kubelet and the runtime to honor the field.
func podSpecUsesHostCgroupNamespace(spec *v1.PodSpec) bool {
	if spec == nil {
		return false
	}
	for i := range spec.InitContainers {
		if spec.InitContainers[i].HostCgroupNamespace != nil {
			return true
		}
	}
	for i := range spec.Containers {
		if spec.Containers[i].HostCgroupNamespace != nil {
			return true
		}
	}
	return false
}
