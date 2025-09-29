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
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/features"
)

// Features carries feature gate values used by various plugins.
// This struct allows us to break the dependency of the plugins on
// the internal k8s features pkg.
type Features struct {
	EnableDRAExtendedResource                    bool
	EnableDRAPrioritizedList                     bool
	EnableDRAAdminAccess                         bool
	EnableDRADeviceTaints                        bool
	EnableDRADeviceBindingConditions             bool
	EnableDRAResourceClaimDeviceStatus           bool
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
	EnableConsumableCapacity                     bool
}

// NewSchedulerFeaturesFromGates copies the current state of the feature gates into the struct.
func NewSchedulerFeaturesFromGates(featureGate featuregate.FeatureGate) Features {
	return Features{
		EnableDRAExtendedResource:                    featureGate.Enabled(features.DRAExtendedResource),
		EnableDRAPrioritizedList:                     featureGate.Enabled(features.DRAPrioritizedList),
		EnableDRAAdminAccess:                         featureGate.Enabled(features.DRAAdminAccess),
		EnableConsumableCapacity:                     featureGate.Enabled(features.DRAConsumableCapacity),
		EnableDRADeviceTaints:                        featureGate.Enabled(features.DRADeviceTaints),
		EnableDRASchedulerFilterTimeout:              featureGate.Enabled(features.DRASchedulerFilterTimeout),
		EnableDRAResourceClaimDeviceStatus:           featureGate.Enabled(features.DRAResourceClaimDeviceStatus),
		EnableDRADeviceBindingConditions:             featureGate.Enabled(features.DRADeviceBindingConditions),
		EnableDynamicResourceAllocation:              featureGate.Enabled(features.DynamicResourceAllocation),
		EnableVolumeAttributesClass:                  featureGate.Enabled(features.VolumeAttributesClass),
		EnableCSIMigrationPortworx:                   featureGate.Enabled(features.CSIMigrationPortworx),
		EnableNodeInclusionPolicyInPodTopologySpread: featureGate.Enabled(features.NodeInclusionPolicyInPodTopologySpread),
		EnableMatchLabelKeysInPodTopologySpread:      featureGate.Enabled(features.MatchLabelKeysInPodTopologySpread),
		EnableInPlacePodVerticalScaling:              featureGate.Enabled(features.InPlacePodVerticalScaling),
		EnableSidecarContainers:                      featureGate.Enabled(features.SidecarContainers),
		EnableSchedulingQueueHint:                    featureGate.Enabled(features.SchedulerQueueingHints),
		EnableAsyncPreemption:                        featureGate.Enabled(features.SchedulerAsyncPreemption),
		EnablePodLevelResources:                      featureGate.Enabled(features.PodLevelResources),
		EnablePartitionableDevices:                   featureGate.Enabled(features.DRAPartitionableDevices),
		EnableStorageCapacityScoring:                 featureGate.Enabled(features.StorageCapacityScoring),
	}
}
