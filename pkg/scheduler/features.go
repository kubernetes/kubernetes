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

package scheduler

import (
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
)

func GetPluginFeatures() plfeature.Features {
	return plfeature.Features{
		EnableCSIMigration:                  feature.DefaultFeatureGate.Enabled(features.CSIMigration),
		EnableCSIMigrationAWS:               feature.DefaultFeatureGate.Enabled(features.CSIMigrationAWS),
		EnableCSIMigrationGCE:               feature.DefaultFeatureGate.Enabled(features.CSIMigrationGCE),
		EnableCSIMigrationAzureDisk:         feature.DefaultFeatureGate.Enabled(features.CSIMigrationAzureDisk),
		EnableCSIMigrationOpenStack:         feature.DefaultFeatureGate.Enabled(features.CSIMigrationOpenStack),
		EnableCSIStorageCapacity:            feature.DefaultFeatureGate.Enabled(features.CSIStorageCapacity),
		EnableDefaultPodTopologySpread:      feature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread),
		EnableGenericEphemeralVolume:        feature.DefaultFeatureGate.Enabled(features.GenericEphemeralVolume),
		EnableLocalStorageCapacityIsolation: feature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation),
		EnablePodAffinityNamespaceSelector:  feature.DefaultFeatureGate.Enabled(features.PodAffinityNamespaceSelector),
		EnablePodDisruptionBudget:           feature.DefaultFeatureGate.Enabled(features.PodDisruptionBudget),
		EnablePodOverhead:                   feature.DefaultFeatureGate.Enabled(features.PodOverhead),
		EnableReadWriteOncePod:              feature.DefaultFeatureGate.Enabled(features.ReadWriteOncePod),
		EnableStorageObjectInUseProtection:  feature.DefaultFeatureGate.Enabled(features.StorageObjectInUseProtection),
		EnableVolumeCapacityPriority:        feature.DefaultFeatureGate.Enabled(features.VolumeCapacityPriority),
	}
}
