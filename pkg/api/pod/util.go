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

package pod

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// Visitor is called with each object name, and returns true if visiting should continue
type Visitor func(name string) (shouldContinue bool)

// VisitPodSecretNames invokes the visitor function with the name of every secret
// referenced by the pod spec. If visitor returns false, visiting is short-circuited.
// Transitive references (e.g. pod -> pvc -> pv -> secret) are not visited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPodSecretNames(pod *api.Pod, visitor Visitor) bool {
	for _, reference := range pod.Spec.ImagePullSecrets {
		if !visitor(reference.Name) {
			return false
		}
	}
	for i := range pod.Spec.InitContainers {
		if !visitContainerSecretNames(&pod.Spec.InitContainers[i], visitor) {
			return false
		}
	}
	for i := range pod.Spec.Containers {
		if !visitContainerSecretNames(&pod.Spec.Containers[i], visitor) {
			return false
		}
	}
	var source *api.VolumeSource
	for i := range pod.Spec.Volumes {
		source = &pod.Spec.Volumes[i].VolumeSource
		switch {
		case source.AzureFile != nil:
			if len(source.AzureFile.SecretName) > 0 && !visitor(source.AzureFile.SecretName) {
				return false
			}
		case source.CephFS != nil:
			if source.CephFS.SecretRef != nil && !visitor(source.CephFS.SecretRef.Name) {
				return false
			}
		case source.Cinder != nil:
			if source.Cinder.SecretRef != nil && !visitor(source.Cinder.SecretRef.Name) {
				return false
			}
		case source.FlexVolume != nil:
			if source.FlexVolume.SecretRef != nil && !visitor(source.FlexVolume.SecretRef.Name) {
				return false
			}
		case source.Projected != nil:
			for j := range source.Projected.Sources {
				if source.Projected.Sources[j].Secret != nil {
					if !visitor(source.Projected.Sources[j].Secret.Name) {
						return false
					}
				}
			}
		case source.RBD != nil:
			if source.RBD.SecretRef != nil && !visitor(source.RBD.SecretRef.Name) {
				return false
			}
		case source.Secret != nil:
			if !visitor(source.Secret.SecretName) {
				return false
			}
		case source.ScaleIO != nil:
			if source.ScaleIO.SecretRef != nil && !visitor(source.ScaleIO.SecretRef.Name) {
				return false
			}
		case source.ISCSI != nil:
			if source.ISCSI.SecretRef != nil && !visitor(source.ISCSI.SecretRef.Name) {
				return false
			}
		case source.StorageOS != nil:
			if source.StorageOS.SecretRef != nil && !visitor(source.StorageOS.SecretRef.Name) {
				return false
			}
		}
	}
	return true
}

func visitContainerSecretNames(container *api.Container, visitor Visitor) bool {
	for _, env := range container.EnvFrom {
		if env.SecretRef != nil {
			if !visitor(env.SecretRef.Name) {
				return false
			}
		}
	}
	for _, envVar := range container.Env {
		if envVar.ValueFrom != nil && envVar.ValueFrom.SecretKeyRef != nil {
			if !visitor(envVar.ValueFrom.SecretKeyRef.Name) {
				return false
			}
		}
	}
	return true
}

// VisitPodConfigmapNames invokes the visitor function with the name of every configmap
// referenced by the pod spec. If visitor returns false, visiting is short-circuited.
// Transitive references (e.g. pod -> pvc -> pv -> secret) are not visited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPodConfigmapNames(pod *api.Pod, visitor Visitor) bool {
	for i := range pod.Spec.InitContainers {
		if !visitContainerConfigmapNames(&pod.Spec.InitContainers[i], visitor) {
			return false
		}
	}
	for i := range pod.Spec.Containers {
		if !visitContainerConfigmapNames(&pod.Spec.Containers[i], visitor) {
			return false
		}
	}
	var source *api.VolumeSource
	for i := range pod.Spec.Volumes {
		source = &pod.Spec.Volumes[i].VolumeSource
		switch {
		case source.Projected != nil:
			for j := range source.Projected.Sources {
				if source.Projected.Sources[j].ConfigMap != nil {
					if !visitor(source.Projected.Sources[j].ConfigMap.Name) {
						return false
					}
				}
			}
		case source.ConfigMap != nil:
			if !visitor(source.ConfigMap.Name) {
				return false
			}
		}
	}
	return true
}

func visitContainerConfigmapNames(container *api.Container, visitor Visitor) bool {
	for _, env := range container.EnvFrom {
		if env.ConfigMapRef != nil {
			if !visitor(env.ConfigMapRef.Name) {
				return false
			}
		}
	}
	for _, envVar := range container.Env {
		if envVar.ValueFrom != nil && envVar.ValueFrom.ConfigMapKeyRef != nil {
			if !visitor(envVar.ValueFrom.ConfigMapKeyRef.Name) {
				return false
			}
		}
	}
	return true
}

// IsPodReady returns true if a pod is ready; false otherwise.
func IsPodReady(pod *api.Pod) bool {
	return IsPodReadyConditionTrue(pod.Status)
}

// IsPodReadyConditionTrue returns true if a pod is ready; false otherwise.
func IsPodReadyConditionTrue(status api.PodStatus) bool {
	condition := GetPodReadyCondition(status)
	return condition != nil && condition.Status == api.ConditionTrue
}

// GetPodReadyCondition extracts the pod ready condition from the given status and returns that.
// Returns nil if the condition is not present.
func GetPodReadyCondition(status api.PodStatus) *api.PodCondition {
	_, condition := GetPodCondition(&status, api.PodReady)
	return condition
}

// GetPodCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetPodCondition(status *api.PodStatus, conditionType api.PodConditionType) (int, *api.PodCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}

// UpdatePodCondition updates existing pod condition or creates a new one. Sets LastTransitionTime to now if the
// status has changed.
// Returns true if pod condition has changed or has been added.
func UpdatePodCondition(status *api.PodStatus, condition *api.PodCondition) bool {
	condition.LastTransitionTime = metav1.Now()
	// Try to find this pod condition.
	conditionIndex, oldCondition := GetPodCondition(status, condition.Type)

	if oldCondition == nil {
		// We are adding new pod condition.
		status.Conditions = append(status.Conditions, *condition)
		return true
	}
	// We are updating an existing condition, so we need to check if it has changed.
	if condition.Status == oldCondition.Status {
		condition.LastTransitionTime = oldCondition.LastTransitionTime
	}

	isEqual := condition.Status == oldCondition.Status &&
		condition.Reason == oldCondition.Reason &&
		condition.Message == oldCondition.Message &&
		condition.LastProbeTime.Equal(&oldCondition.LastProbeTime) &&
		condition.LastTransitionTime.Equal(&oldCondition.LastTransitionTime)

	status.Conditions[conditionIndex] = *condition
	// Return true if one of the fields have changed.
	return !isEqual
}

// DropDisabledFields removes disabled fields from the pod spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pod spec.
func DropDisabledFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodPriority) {
		podSpec.Priority = nil
		podSpec.PriorityClassName = ""
		if oldPodSpec != nil {
			oldPodSpec.Priority = nil
			oldPodSpec.PriorityClassName = ""
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		for i := range podSpec.Volumes {
			if podSpec.Volumes[i].EmptyDir != nil {
				podSpec.Volumes[i].EmptyDir.SizeLimit = nil
			}
		}
		if oldPodSpec != nil {
			for i := range oldPodSpec.Volumes {
				if oldPodSpec.Volumes[i].EmptyDir != nil {
					oldPodSpec.Volumes[i].EmptyDir.SizeLimit = nil
				}
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) && !subpathInUse(oldPodSpec) {
		// drop subpath from the pod if the feature is disabled and the old spec did not specify subpaths
		for i := range podSpec.Containers {
			for j := range podSpec.Containers[i].VolumeMounts {
				podSpec.Containers[i].VolumeMounts[j].SubPath = ""
			}
		}
		for i := range podSpec.InitContainers {
			for j := range podSpec.InitContainers[i].VolumeMounts {
				podSpec.InitContainers[i].VolumeMounts[j].SubPath = ""
			}
		}
	}

	dropDisabledVolumeDevicesAlphaFields(podSpec, oldPodSpec)

	dropDisabledRunAsGroupField(podSpec, oldPodSpec)

	if !utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClass) && !runtimeClassInUse(oldPodSpec) {
		// Set RuntimeClassName to nil only if feature is disabled and it is not used
		podSpec.RuntimeClassName = nil
	}

	dropDisabledProcMountField(podSpec, oldPodSpec)
}

// dropDisabledRunAsGroupField removes disabled fields from PodSpec related
// to RunAsGroup
func dropDisabledRunAsGroupField(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) {
		if podSpec.SecurityContext != nil {
			podSpec.SecurityContext.RunAsGroup = nil
		}
		for i := range podSpec.Containers {
			if podSpec.Containers[i].SecurityContext != nil {
				podSpec.Containers[i].SecurityContext.RunAsGroup = nil
			}
		}
		for i := range podSpec.InitContainers {
			if podSpec.InitContainers[i].SecurityContext != nil {
				podSpec.InitContainers[i].SecurityContext.RunAsGroup = nil
			}
		}

		if oldPodSpec != nil {
			if oldPodSpec.SecurityContext != nil {
				oldPodSpec.SecurityContext.RunAsGroup = nil
			}
			for i := range oldPodSpec.Containers {
				if oldPodSpec.Containers[i].SecurityContext != nil {
					oldPodSpec.Containers[i].SecurityContext.RunAsGroup = nil
				}
			}
			for i := range oldPodSpec.InitContainers {
				if oldPodSpec.InitContainers[i].SecurityContext != nil {
					oldPodSpec.InitContainers[i].SecurityContext.RunAsGroup = nil
				}
			}
		}
	}
}

// dropDisabledProcMountField removes disabled fields from PodSpec related
// to ProcMount only if it is not already used by the old spec
func dropDisabledProcMountField(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) && !procMountInUse(oldPodSpec) {
		defaultProcMount := api.DefaultProcMount
		for i := range podSpec.Containers {
			if podSpec.Containers[i].SecurityContext != nil {
				podSpec.Containers[i].SecurityContext.ProcMount = &defaultProcMount
			}
		}
		for i := range podSpec.InitContainers {
			if podSpec.InitContainers[i].SecurityContext != nil {
				podSpec.InitContainers[i].SecurityContext.ProcMount = &defaultProcMount
			}
		}
	}
}

// dropDisabledVolumeDevicesAlphaFields removes disabled fields from []VolumeDevice.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a VolumeDevice
func dropDisabledVolumeDevicesAlphaFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		for i := range podSpec.Containers {
			podSpec.Containers[i].VolumeDevices = nil
		}
		for i := range podSpec.InitContainers {
			podSpec.InitContainers[i].VolumeDevices = nil
		}

		if oldPodSpec != nil {
			for i := range oldPodSpec.Containers {
				oldPodSpec.Containers[i].VolumeDevices = nil
			}
			for i := range oldPodSpec.InitContainers {
				oldPodSpec.InitContainers[i].VolumeDevices = nil
			}
		}
	}
}

// subpathInUse returns true if the pod spec is non-nil and has a volume mount that makes use of the subPath feature
func subpathInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for i := range podSpec.Containers {
		for j := range podSpec.Containers[i].VolumeMounts {
			if len(podSpec.Containers[i].VolumeMounts[j].SubPath) > 0 {
				return true
			}
		}
	}
	for i := range podSpec.InitContainers {
		for j := range podSpec.InitContainers[i].VolumeMounts {
			if len(podSpec.InitContainers[i].VolumeMounts[j].SubPath) > 0 {
				return true
			}
		}
	}
	return false
}

// runtimeClassInUse returns true if the pod spec is non-nil and has a RuntimeClassName set
func runtimeClassInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	if podSpec.RuntimeClassName != nil {
		return true
	}
	return false
}

// procMountInUse returns true if the pod spec is non-nil and has a SecurityContext's ProcMount field set
func procMountInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].SecurityContext != nil {
			if podSpec.Containers[i].SecurityContext.ProcMount != nil {
				if *podSpec.Containers[i].SecurityContext.ProcMount != api.DefaultProcMount {
					return true
				}
			}
		}
	}
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].SecurityContext != nil {
			if podSpec.InitContainers[i].SecurityContext.ProcMount != nil {
				if *podSpec.InitContainers[i].SecurityContext.ProcMount != api.DefaultProcMount {
					return true
				}
			}
		}
	}
	return false
}
