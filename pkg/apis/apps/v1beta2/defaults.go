/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta2

import (
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_DaemonSet(obj *appsv1beta2.DaemonSet) {
	updateStrategy := &obj.Spec.UpdateStrategy
	if updateStrategy.Type == "" {
		updateStrategy.Type = appsv1beta2.RollingUpdateDaemonSetStrategyType
	}
	if updateStrategy.Type == appsv1beta2.RollingUpdateDaemonSetStrategyType {
		if updateStrategy.RollingUpdate == nil {
			rollingUpdate := appsv1beta2.RollingUpdateDaemonSet{}
			updateStrategy.RollingUpdate = &rollingUpdate
		}
		if updateStrategy.RollingUpdate.MaxUnavailable == nil {
			// Set default MaxUnavailable as 1 by default.
			maxUnavailable := intstr.FromInt(1)
			updateStrategy.RollingUpdate.MaxUnavailable = &maxUnavailable
		}
		if updateStrategy.RollingUpdate.MaxSurge == nil {
			// Set default MaxSurge as 0 by default.
			maxSurge := intstr.FromInt(0)
			updateStrategy.RollingUpdate.MaxSurge = &maxSurge
		}
	}
	if obj.Spec.RevisionHistoryLimit == nil {
		obj.Spec.RevisionHistoryLimit = new(int32)
		*obj.Spec.RevisionHistoryLimit = 10
	}
}

func SetDefaults_StatefulSet(obj *appsv1beta2.StatefulSet) {
	if len(obj.Spec.PodManagementPolicy) == 0 {
		obj.Spec.PodManagementPolicy = appsv1beta2.OrderedReadyPodManagement
	}

	if obj.Spec.UpdateStrategy.Type == "" {
		obj.Spec.UpdateStrategy.Type = appsv1beta2.RollingUpdateStatefulSetStrategyType

		if obj.Spec.UpdateStrategy.RollingUpdate == nil {
			// UpdateStrategy.RollingUpdate will take default values below.
			obj.Spec.UpdateStrategy.RollingUpdate = &appsv1beta2.RollingUpdateStatefulSetStrategy{}
		}
	}

	if obj.Spec.UpdateStrategy.Type == appsv1beta2.RollingUpdateStatefulSetStrategyType &&
		obj.Spec.UpdateStrategy.RollingUpdate != nil {

		if obj.Spec.UpdateStrategy.RollingUpdate.Partition == nil {
			obj.Spec.UpdateStrategy.RollingUpdate.Partition = pointer.Int32(0)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.MaxUnavailableStatefulSet) {
			if obj.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable == nil {
				maxUnavailable := intstr.FromInt(1)
				obj.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable = &maxUnavailable
			}
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.StatefulSetAutoDeletePVC) {
		if obj.Spec.PersistentVolumeClaimRetentionPolicy == nil {
			obj.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1beta2.StatefulSetPersistentVolumeClaimRetentionPolicy{}
		}
		if len(obj.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted) == 0 {
			obj.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted = appsv1beta2.RetainPersistentVolumeClaimRetentionPolicyType
		}
		if len(obj.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled) == 0 {
			obj.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled = appsv1beta2.RetainPersistentVolumeClaimRetentionPolicyType
		}
	}

	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
	if obj.Spec.RevisionHistoryLimit == nil {
		obj.Spec.RevisionHistoryLimit = new(int32)
		*obj.Spec.RevisionHistoryLimit = 10
	}
}

// SetDefaults_Deployment sets additional defaults compared to its counterpart
// in extensions. These addons are:
// - MaxUnavailable during rolling update set to 25% (1 in extensions)
// - MaxSurge value during rolling update set to 25% (1 in extensions)
// - RevisionHistoryLimit set to 10 (not set in extensions)
// - ProgressDeadlineSeconds set to 600s (not set in extensions)
func SetDefaults_Deployment(obj *appsv1beta2.Deployment) {
	// Set appsv1beta2.DeploymentSpec.Replicas to 1 if it is not set.
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
	strategy := &obj.Spec.Strategy
	// Set default appsv1beta2.DeploymentStrategyType as RollingUpdate.
	if strategy.Type == "" {
		strategy.Type = appsv1beta2.RollingUpdateDeploymentStrategyType
	}
	if strategy.Type == appsv1beta2.RollingUpdateDeploymentStrategyType {
		if strategy.RollingUpdate == nil {
			rollingUpdate := appsv1beta2.RollingUpdateDeployment{}
			strategy.RollingUpdate = &rollingUpdate
		}
		if strategy.RollingUpdate.MaxUnavailable == nil {
			// Set default MaxUnavailable as 25% by default.
			maxUnavailable := intstr.FromString("25%")
			strategy.RollingUpdate.MaxUnavailable = &maxUnavailable
		}
		if strategy.RollingUpdate.MaxSurge == nil {
			// Set default MaxSurge as 25% by default.
			maxSurge := intstr.FromString("25%")
			strategy.RollingUpdate.MaxSurge = &maxSurge
		}
	}
	if obj.Spec.RevisionHistoryLimit == nil {
		obj.Spec.RevisionHistoryLimit = new(int32)
		*obj.Spec.RevisionHistoryLimit = 10
	}
	if obj.Spec.ProgressDeadlineSeconds == nil {
		obj.Spec.ProgressDeadlineSeconds = new(int32)
		*obj.Spec.ProgressDeadlineSeconds = 600
	}
}

func SetDefaults_ReplicaSet(obj *appsv1beta2.ReplicaSet) {
	if obj.Spec.Replicas == nil {
		obj.Spec.Replicas = new(int32)
		*obj.Spec.Replicas = 1
	}
}
