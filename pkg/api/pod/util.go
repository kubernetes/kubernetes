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
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// ContainerType signifies container type
type ContainerType int

const (
	// Containers is for normal containers
	Containers ContainerType = 1 << iota
	// InitContainers is for init containers
	InitContainers
	// EphemeralContainers is for ephemeral containers
	EphemeralContainers
)

// AllContainers specifies that all containers be visited
const AllContainers ContainerType = (InitContainers | Containers | EphemeralContainers)

// AllFeatureEnabledContainers returns a ContainerType mask which includes all container
// types except for the ones guarded by feature gate.
func AllFeatureEnabledContainers() ContainerType {
	containerType := AllContainers
	if !utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) {
		containerType &= ^EphemeralContainers
	}
	return containerType
}

// ContainerVisitor is called with each container spec, and returns true
// if visiting should continue.
type ContainerVisitor func(container *api.Container, containerType ContainerType) (shouldContinue bool)

// VisitContainers invokes the visitor function with a pointer to every container
// spec in the given pod spec with type set in mask. If visitor returns false,
// visiting is short-circuited. VisitContainers returns true if visiting completes,
// false if visiting was short-circuited.
func VisitContainers(podSpec *api.PodSpec, mask ContainerType, visitor ContainerVisitor) bool {
	if mask&InitContainers != 0 {
		for i := range podSpec.InitContainers {
			if !visitor(&podSpec.InitContainers[i], InitContainers) {
				return false
			}
		}
	}
	if mask&Containers != 0 {
		for i := range podSpec.Containers {
			if !visitor(&podSpec.Containers[i], Containers) {
				return false
			}
		}
	}
	if mask&EphemeralContainers != 0 {
		for i := range podSpec.EphemeralContainers {
			if !visitor((*api.Container)(&podSpec.EphemeralContainers[i].EphemeralContainerCommon), EphemeralContainers) {
				return false
			}
		}
	}
	return true
}

// Visitor is called with each object name, and returns true if visiting should continue
type Visitor func(name string) (shouldContinue bool)

// VisitPodSecretNames invokes the visitor function with the name of every secret
// referenced by the pod spec. If visitor returns false, visiting is short-circuited.
// Transitive references (e.g. pod -> pvc -> pv -> secret) are not visited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPodSecretNames(pod *api.Pod, visitor Visitor, containerType ContainerType) bool {
	for _, reference := range pod.Spec.ImagePullSecrets {
		if !visitor(reference.Name) {
			return false
		}
	}
	VisitContainers(&pod.Spec, containerType, func(c *api.Container, containerType ContainerType) bool {
		return visitContainerSecretNames(c, visitor)
	})
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
		case source.CSI != nil:
			if source.CSI.NodePublishSecretRef != nil && !visitor(source.CSI.NodePublishSecretRef.Name) {
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
func VisitPodConfigmapNames(pod *api.Pod, visitor Visitor, containerType ContainerType) bool {
	VisitContainers(&pod.Spec, containerType, func(c *api.Container, containerType ContainerType) bool {
		return visitContainerConfigmapNames(c, visitor)
	})
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

// DropDisabledTemplateFields removes disabled fields from the pod template metadata and spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a PodTemplateSpec
func DropDisabledTemplateFields(podTemplate, oldPodTemplate *api.PodTemplateSpec) {
	var (
		podSpec           *api.PodSpec
		podAnnotations    map[string]string
		oldPodSpec        *api.PodSpec
		oldPodAnnotations map[string]string
	)
	if podTemplate != nil {
		podSpec = &podTemplate.Spec
		podAnnotations = podTemplate.Annotations
	}
	if oldPodTemplate != nil {
		oldPodSpec = &oldPodTemplate.Spec
		oldPodAnnotations = oldPodTemplate.Annotations
	}
	dropDisabledFields(podSpec, podAnnotations, oldPodSpec, oldPodAnnotations)
}

// DropDisabledPodFields removes disabled fields from the pod metadata and spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a Pod
func DropDisabledPodFields(pod, oldPod *api.Pod) {
	var (
		podSpec           *api.PodSpec
		podAnnotations    map[string]string
		oldPodSpec        *api.PodSpec
		oldPodAnnotations map[string]string
		podStatus         *api.PodStatus
		oldPodStatus      *api.PodStatus
	)
	if pod != nil {
		podSpec = &pod.Spec
		podAnnotations = pod.Annotations
		podStatus = &pod.Status
	}
	if oldPod != nil {
		oldPodSpec = &oldPod.Spec
		oldPodAnnotations = oldPod.Annotations
		oldPodStatus = &oldPod.Status
	}
	dropDisabledFields(podSpec, podAnnotations, oldPodSpec, oldPodAnnotations)
	dropPodStatusDisabledFields(podStatus, oldPodStatus)
}

// dropPodStatusDisabledFields removes disabled fields from the pod status
func dropPodStatusDisabledFields(podStatus *api.PodStatus, oldPodStatus *api.PodStatus) {
	// trim PodIPs down to only one entry (non dual stack).
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) &&
		!multiplePodIPsInUse(oldPodStatus) {
		if len(podStatus.PodIPs) != 0 {
			podStatus.PodIPs = podStatus.PodIPs[0:1]
		}
	}
}

// dropDisabledFields removes disabled fields from the pod metadata and spec.
func dropDisabledFields(
	podSpec *api.PodSpec, podAnnotations map[string]string,
	oldPodSpec *api.PodSpec, oldPodAnnotations map[string]string,
) {
	// the new spec must always be non-nil
	if podSpec == nil {
		podSpec = &api.PodSpec{}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequestProjection) &&
		!tokenRequestProjectionInUse(oldPodSpec) {
		for i := range podSpec.Volumes {
			if podSpec.Volumes[i].Projected != nil {
				for j := range podSpec.Volumes[i].Projected.Sources {
					podSpec.Volumes[i].Projected.Sources[j].ServiceAccountToken = nil
				}
			}

		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.AppArmor) && !appArmorInUse(oldPodAnnotations) {
		for k := range podAnnotations {
			if strings.HasPrefix(k, v1.AppArmorBetaContainerAnnotationKeyPrefix) {
				delete(podAnnotations, k)
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.Sysctls) && !sysctlsInUse(oldPodSpec) {
		if podSpec.SecurityContext != nil {
			podSpec.SecurityContext.Sysctls = nil
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) && !emptyDirSizeLimitInUse(oldPodSpec) {
		for i := range podSpec.Volumes {
			if podSpec.Volumes[i].EmptyDir != nil {
				podSpec.Volumes[i].EmptyDir.SizeLimit = nil
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) && !subpathInUse(oldPodSpec) {
		// drop subpath from the pod if the feature is disabled and the old spec did not specify subpaths
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			for i := range c.VolumeMounts {
				c.VolumeMounts[i].SubPath = ""
			}
			return true
		})
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) && !ephemeralContainersInUse(oldPodSpec) {
		podSpec.EphemeralContainers = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) && !subpathExprInUse(oldPodSpec) {
		// drop subpath env expansion from the pod if subpath feature is disabled and the old spec did not specify subpath env expansion
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			for i := range c.VolumeMounts {
				c.VolumeMounts[i].SubPathExpr = ""
			}
			return true
		})
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.StartupProbe) && !startupProbeInUse(oldPodSpec) {
		// drop startupProbe from all containers if the feature is disabled
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			c.StartupProbe = nil
			return true
		})
	}

	dropDisabledRunAsGroupField(podSpec, oldPodSpec)

	dropDisabledFSGroupFields(podSpec, oldPodSpec)

	if !utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClass) && !runtimeClassInUse(oldPodSpec) {
		// Set RuntimeClassName to nil only if feature is disabled and it is not used
		podSpec.RuntimeClassName = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.PodOverhead) && !overheadInUse(oldPodSpec) {
		// Set Overhead to nil only if the feature is disabled and it is not used
		podSpec.Overhead = nil
	}

	dropDisabledProcMountField(podSpec, oldPodSpec)

	dropDisabledCSIVolumeSourceAlphaFields(podSpec, oldPodSpec)

	if !utilfeature.DefaultFeatureGate.Enabled(features.NonPreemptingPriority) &&
		!podPriorityInUse(oldPodSpec) {
		// Set to nil pod's PreemptionPolicy fields if the feature is disabled and the old pod
		// does not specify any values for these fields.
		podSpec.PreemptionPolicy = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.EvenPodsSpread) && !topologySpreadConstraintsInUse(oldPodSpec) {
		// Set TopologySpreadConstraints to nil only if feature is disabled and it is not used
		podSpec.TopologySpreadConstraints = nil
	}
}

// dropDisabledRunAsGroupField removes disabled fields from PodSpec related
// to RunAsGroup
func dropDisabledRunAsGroupField(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) && !runAsGroupInUse(oldPodSpec) {
		if podSpec.SecurityContext != nil {
			podSpec.SecurityContext.RunAsGroup = nil
		}
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			if c.SecurityContext != nil {
				c.SecurityContext.RunAsGroup = nil
			}
			return true
		})
	}
}

// dropDisabledProcMountField removes disabled fields from PodSpec related
// to ProcMount only if it is not already used by the old spec
func dropDisabledProcMountField(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) && !procMountInUse(oldPodSpec) {
		defaultProcMount := api.DefaultProcMount
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			if c.SecurityContext != nil && c.SecurityContext.ProcMount != nil {
				// The ProcMount field was improperly forced to non-nil in 1.12.
				// If the feature is disabled, and the existing object is not using any non-default values, and the ProcMount field is present in the incoming object, force to the default value.
				// Note: we cannot force the field to nil when the feature is disabled because it causes a diff against previously persisted data.
				c.SecurityContext.ProcMount = &defaultProcMount
			}
			return true
		})
	}
}

func dropDisabledFSGroupFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ConfigurableFSGroupPolicy) && !fsGroupPolicyInUse(oldPodSpec) {
		// if oldPodSpec had no FSGroupChangePolicy set then we should prevent new pod from having this field
		// if ConfigurableFSGroupPolicy feature is disabled
		if podSpec.SecurityContext != nil {
			podSpec.SecurityContext.FSGroupChangePolicy = nil
		}
	}
}

// dropDisabledCSIVolumeSourceAlphaFields removes disabled alpha fields from []CSIVolumeSource.
// This should be called from PrepareForCreate/PrepareForUpdate for all pod specs resources containing a CSIVolumeSource
func dropDisabledCSIVolumeSourceAlphaFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) && !csiInUse(oldPodSpec) {
		for i := range podSpec.Volumes {
			podSpec.Volumes[i].CSI = nil
		}
	}
}

func ephemeralContainersInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	return len(podSpec.EphemeralContainers) > 0
}

func fsGroupPolicyInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	securityContext := podSpec.SecurityContext
	if securityContext != nil && securityContext.FSGroupChangePolicy != nil {
		return true
	}
	return false
}

// subpathInUse returns true if the pod spec is non-nil and has a volume mount that makes use of the subPath feature
func subpathInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		for i := range c.VolumeMounts {
			if len(c.VolumeMounts[i].SubPath) > 0 {
				inUse = true
				return false
			}
		}
		return true
	})

	return inUse
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

// overheadInUse returns true if the pod spec is non-nil and has Overhead set
func overheadInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	if podSpec.Overhead != nil {
		return true
	}
	return false
}

// topologySpreadConstraintsInUse returns true if the pod spec is non-nil and has a TopologySpreadConstraints slice
func topologySpreadConstraintsInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	return len(podSpec.TopologySpreadConstraints) > 0
}

// procMountInUse returns true if the pod spec is non-nil and has a SecurityContext's ProcMount field set to a non-default value
func procMountInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		if c.SecurityContext == nil || c.SecurityContext.ProcMount == nil {
			return true
		}
		if *c.SecurityContext.ProcMount != api.DefaultProcMount {
			inUse = true
			return false
		}
		return true
	})

	return inUse
}

// appArmorInUse returns true if the pod has apparmor related information
func appArmorInUse(podAnnotations map[string]string) bool {
	for k := range podAnnotations {
		if strings.HasPrefix(k, v1.AppArmorBetaContainerAnnotationKeyPrefix) {
			return true
		}
	}
	return false
}

func tokenRequestProjectionInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for _, v := range podSpec.Volumes {
		if v.Projected == nil {
			continue
		}
		for _, s := range v.Projected.Sources {
			if s.ServiceAccountToken != nil {
				return true
			}
		}
	}
	return false
}

// podPriorityInUse returns true if the pod spec is non-nil and has Priority or PriorityClassName set.
func podPriorityInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	if podSpec.Priority != nil || podSpec.PriorityClassName != "" {
		return true
	}
	return false
}

func sysctlsInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	if podSpec.SecurityContext != nil && podSpec.SecurityContext.Sysctls != nil {
		return true
	}
	return false
}

// emptyDirSizeLimitInUse returns true if any pod's EmptyDir volumes use SizeLimit.
func emptyDirSizeLimitInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].EmptyDir != nil {
			if podSpec.Volumes[i].EmptyDir.SizeLimit != nil {
				return true
			}
		}
	}
	return false
}

// runAsGroupInUse returns true if the pod spec is non-nil and has a SecurityContext's RunAsGroup field set
func runAsGroupInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.RunAsGroup != nil {
		return true
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		if c.SecurityContext != nil && c.SecurityContext.RunAsGroup != nil {
			inUse = true
			return false
		}
		return true
	})

	return inUse
}

// subpathExprInUse returns true if the pod spec is non-nil and has a volume mount that makes use of the subPathExpr feature
func subpathExprInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		for i := range c.VolumeMounts {
			if len(c.VolumeMounts[i].SubPathExpr) > 0 {
				inUse = true
				return false
			}
		}
		return true
	})

	return inUse
}

// startupProbeInUse returns true if the pod spec is non-nil and has a container that has a startupProbe defined
func startupProbeInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		if c.StartupProbe != nil {
			inUse = true
			return false
		}
		return true
	})

	return inUse
}

// csiInUse returns true if any pod's spec include inline CSI volumes.
func csiInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].CSI != nil {
			return true
		}
	}
	return false
}

// podPriorityInUse returns true if status is not nil and number of PodIPs is greater than one
func multiplePodIPsInUse(podStatus *api.PodStatus) bool {
	if podStatus == nil {
		return false
	}
	if len(podStatus.PodIPs) > 1 {
		return true
	}
	return false
}
