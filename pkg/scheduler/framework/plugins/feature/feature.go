/*
Copyright 2021 The Kubernetes Authors.

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

package feature

import (
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// Features carries feature gate values used by various plugins.
// This struct allows us to break the dependency of the plugins on
// the internal k8s features pkg.
type Features struct {
	EnableDRAPrioritizedList                     bool
	EnableDRAAdminAccess                         bool
	EnableDRADeviceTaints                        bool
	EnableDRASchedulerFilterTimeout              bool
	EnableDynamicResourceAllocation              bool
	EnableVolumeAttributesClass                  bool
	EnableCSIMigrationPortworx                   bool
	EnableNodeInclusionPolicyInPodTopologySpread bool
	EnableMatchLabelKeysInPodTopologySpread      bool
	EnableInPlacePodVerticalScaling              bool
	EnableSidecarContainers                      bool
	EnableSchedulingQueueHint                    bool
	EnableAsyncPreemption                        bool
	EnablePodLevelResources                      bool
	EnablePartitionableDevices                   bool
	EnableStorageCapacityScoring                 bool
}

// NewFeatures copies the current state of the feature gates into the struct.
func NewFeatures() Features {
	return Features{
		EnableDRAPrioritizedList:                     feature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList),
		EnableDRAAdminAccess:                         feature.DefaultFeatureGate.Enabled(features.DRAAdminAccess),
		EnableDRADeviceTaints:                        feature.DefaultFeatureGate.Enabled(features.DRADeviceTaints),
		EnableDRASchedulerFilterTimeout:              feature.DefaultFeatureGate.Enabled(features.DRASchedulerFilterTimeout),
		EnableDynamicResourceAllocation:              feature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation),
		EnableVolumeAttributesClass:                  feature.DefaultFeatureGate.Enabled(features.VolumeAttributesClass),
		EnableCSIMigrationPortworx:                   feature.DefaultFeatureGate.Enabled(features.CSIMigrationPortworx),
		EnableNodeInclusionPolicyInPodTopologySpread: feature.DefaultFeatureGate.Enabled(features.NodeInclusionPolicyInPodTopologySpread),
		EnableMatchLabelKeysInPodTopologySpread:      feature.DefaultFeatureGate.Enabled(features.MatchLabelKeysInPodTopologySpread),
		EnableInPlacePodVerticalScaling:              feature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling),
		EnableSidecarContainers:                      feature.DefaultFeatureGate.Enabled(features.SidecarContainers),
		EnableSchedulingQueueHint:                    feature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints),
		EnableAsyncPreemption:                        feature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption),
		EnablePodLevelResources:                      feature.DefaultFeatureGate.Enabled(features.PodLevelResources),
		EnablePartitionableDevices:                   feature.DefaultFeatureGate.Enabled(features.DRAPartitionableDevices),
		EnableStorageCapacityScoring:                 feature.DefaultFeatureGate.Enabled(features.StorageCapacityScoring),
	}
}
