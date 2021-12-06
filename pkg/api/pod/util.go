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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
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

func skipEmptyNames(visitor Visitor) Visitor {
	return func(name string) bool {
		if len(name) == 0 {
			// continue visiting
			return true
		}
		// delegate to visitor
		return visitor(name)
	}
}

// VisitPodSecretNames invokes the visitor function with the name of every secret
// referenced by the pod spec. If visitor returns false, visiting is short-circuited.
// Transitive references (e.g. pod -> pvc -> pv -> secret) are not visited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPodSecretNames(pod *api.Pod, visitor Visitor, containerType ContainerType) bool {
	visitor = skipEmptyNames(visitor)
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
	visitor = skipEmptyNames(visitor)
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

// usesHugePagesInProjectedVolume returns true if hugepages are used in downward api for volume
func usesHugePagesInProjectedVolume(podSpec *api.PodSpec) bool {
	// determine if any container is using hugepages in downward api volume
	for _, volumeSource := range podSpec.Volumes {
		if volumeSource.DownwardAPI != nil {
			for _, item := range volumeSource.DownwardAPI.Items {
				if item.ResourceFieldRef != nil {
					if strings.HasPrefix(item.ResourceFieldRef.Resource, "requests.hugepages-") || strings.HasPrefix(item.ResourceFieldRef.Resource, "limits.hugepages-") {
						return true
					}
				}
			}
		}
	}
	return false
}

// usesHugePagesInProjectedEnv returns true if hugepages are used in downward api for volume
func usesHugePagesInProjectedEnv(item api.Container) bool {
	for _, env := range item.Env {
		if env.ValueFrom != nil {
			if env.ValueFrom.ResourceFieldRef != nil {
				if strings.HasPrefix(env.ValueFrom.ResourceFieldRef.Resource, "requests.hugepages-") || strings.HasPrefix(env.ValueFrom.ResourceFieldRef.Resource, "limits.hugepages-") {
					return true
				}
			}
		}
	}
	return false
}

func checkContainerUseIndivisibleHugePagesValues(container api.Container) bool {
	for resourceName, quantity := range container.Resources.Limits {
		if helper.IsHugePageResourceName(resourceName) {
			if !helper.IsHugePageResourceValueDivisible(resourceName, quantity) {
				return true
			}
		}
	}

	for resourceName, quantity := range container.Resources.Requests {
		if helper.IsHugePageResourceName(resourceName) {
			if !helper.IsHugePageResourceValueDivisible(resourceName, quantity) {
				return true
			}
		}
	}

	return false
}

// usesIndivisibleHugePagesValues returns true if the one of the containers uses non-integer multiple
// of huge page unit size
func usesIndivisibleHugePagesValues(podSpec *api.PodSpec) bool {
	foundIndivisibleHugePagesValue := false
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		if checkContainerUseIndivisibleHugePagesValues(*c) {
			foundIndivisibleHugePagesValue = true
		}
		return !foundIndivisibleHugePagesValue // continue visiting if we haven't seen an invalid value yet
	})

	if foundIndivisibleHugePagesValue {
		return true
	}

	for resourceName, quantity := range podSpec.Overhead {
		if helper.IsHugePageResourceName(resourceName) {
			if !helper.IsHugePageResourceValueDivisible(resourceName, quantity) {
				return true
			}
		}
	}

	return false
}

// haveSameExpandedDNSConfig returns true if the oldPodSpec already had
// ExpandedDNSConfig and podSpec has the same DNSConfig
func haveSameExpandedDNSConfig(podSpec, oldPodSpec *api.PodSpec) bool {
	if oldPodSpec == nil || oldPodSpec.DNSConfig == nil {
		return false
	}
	if podSpec == nil || podSpec.DNSConfig == nil {
		return false
	}

	if len(oldPodSpec.DNSConfig.Searches) <= apivalidation.MaxDNSSearchPathsLegacy &&
		len(strings.Join(oldPodSpec.DNSConfig.Searches, " ")) <= apivalidation.MaxDNSSearchListCharsLegacy {
		// didn't have ExpandedDNSConfig
		return false
	}

	if len(oldPodSpec.DNSConfig.Searches) != len(podSpec.DNSConfig.Searches) {
		// updates DNSConfig
		return false
	}

	for i, oldSearch := range oldPodSpec.DNSConfig.Searches {
		if podSpec.DNSConfig.Searches[i] != oldSearch {
			// updates DNSConfig
			return false
		}
	}

	return true
}

// GetValidationOptionsFromPodSpecAndMeta returns validation options based on pod specs and metadata
func GetValidationOptionsFromPodSpecAndMeta(podSpec, oldPodSpec *api.PodSpec, podMeta, oldPodMeta *metav1.ObjectMeta) apivalidation.PodValidationOptions {
	// default pod validation options based on feature gate
	opts := apivalidation.PodValidationOptions{
		// Allow pod spec to use hugepages in downward API if feature is enabled
		AllowDownwardAPIHugePages:   utilfeature.DefaultFeatureGate.Enabled(features.DownwardAPIHugePages),
		AllowInvalidPodDeletionCost: !utilfeature.DefaultFeatureGate.Enabled(features.PodDeletionCost),
		// Do not allow pod spec to use non-integer multiple of huge page unit size default
		AllowIndivisibleHugePagesValues: false,
		AllowWindowsHostProcessField:    utilfeature.DefaultFeatureGate.Enabled(features.WindowsHostProcessContainers),
		// Allow pod spec with expanded DNS configuration
		AllowExpandedDNSConfig: utilfeature.DefaultFeatureGate.Enabled(features.ExpandedDNSConfig) || haveSameExpandedDNSConfig(podSpec, oldPodSpec),
		// Allow pod spec to use OS field
		AllowOSField: utilfeature.DefaultFeatureGate.Enabled(features.IdentifyPodOS),
	}

	if oldPodSpec != nil {
		// if old spec used hugepages in downward api, we must allow it
		opts.AllowDownwardAPIHugePages = opts.AllowDownwardAPIHugePages || usesHugePagesInProjectedVolume(oldPodSpec)
		// determine if any container is using hugepages in env var
		if !opts.AllowDownwardAPIHugePages {
			VisitContainers(oldPodSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
				opts.AllowDownwardAPIHugePages = opts.AllowDownwardAPIHugePages || usesHugePagesInProjectedEnv(*c)
				return !opts.AllowDownwardAPIHugePages
			})
		}
		// if old spec has Windows Host Process fields set, we must allow it
		opts.AllowWindowsHostProcessField = opts.AllowWindowsHostProcessField || setsWindowsHostProcess(oldPodSpec)

		// if old spec has OS field set, we must allow it
		opts.AllowOSField = opts.AllowOSField || oldPodSpec.OS != nil

		// if old spec used non-integer multiple of huge page unit size, we must allow it
		opts.AllowIndivisibleHugePagesValues = usesIndivisibleHugePagesValues(oldPodSpec)

	}
	if oldPodMeta != nil && !opts.AllowInvalidPodDeletionCost {
		// This is an update, so validate only if the existing object was valid.
		_, err := helper.GetDeletionCostFromPodAnnotations(oldPodMeta.Annotations)
		opts.AllowInvalidPodDeletionCost = err != nil
	}

	return opts
}

// GetValidationOptionsFromPodTemplate will return pod validation options for specified template.
func GetValidationOptionsFromPodTemplate(podTemplate, oldPodTemplate *api.PodTemplateSpec) apivalidation.PodValidationOptions {
	var newPodSpec, oldPodSpec *api.PodSpec
	var newPodMeta, oldPodMeta *metav1.ObjectMeta
	// we have to be careful about nil pointers here
	// replication controller in particular is prone to passing nil
	if podTemplate != nil {
		newPodSpec = &podTemplate.Spec
		newPodMeta = &podTemplate.ObjectMeta
	}
	if oldPodTemplate != nil {
		oldPodSpec = &oldPodTemplate.Spec
		oldPodMeta = &oldPodTemplate.ObjectMeta
	}
	return GetValidationOptionsFromPodSpecAndMeta(newPodSpec, oldPodSpec, newPodMeta, oldPodMeta)
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
	)
	if pod != nil {
		podSpec = &pod.Spec
		podAnnotations = pod.Annotations
	}
	if oldPod != nil {
		oldPodSpec = &oldPod.Spec
		oldPodAnnotations = oldPod.Annotations
	}
	dropDisabledFields(podSpec, podAnnotations, oldPodSpec, oldPodAnnotations)
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

	if !utilfeature.DefaultFeatureGate.Enabled(features.AppArmor) && !appArmorInUse(oldPodAnnotations) {
		for k := range podAnnotations {
			if strings.HasPrefix(k, v1.AppArmorBetaContainerAnnotationKeyPrefix) {
				delete(podAnnotations, k)
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) && !emptyDirSizeLimitInUse(oldPodSpec) {
		for i := range podSpec.Volumes {
			if podSpec.Volumes[i].EmptyDir != nil {
				podSpec.Volumes[i].EmptyDir.SizeLimit = nil
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) && !ephemeralContainersInUse(oldPodSpec) {
		podSpec.EphemeralContainers = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.ProbeTerminationGracePeriod) && !probeGracePeriodInUse(oldPodSpec) {
		// Set pod-level terminationGracePeriodSeconds to nil if the feature is disabled and it is not used
		VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
			if c.LivenessProbe != nil {
				c.LivenessProbe.TerminationGracePeriodSeconds = nil
			}
			if c.StartupProbe != nil {
				c.StartupProbe.TerminationGracePeriodSeconds = nil
			}
			// cannot be set for readiness probes
			return true
		})
	}

	dropDisabledProcMountField(podSpec, oldPodSpec)

	dropDisabledCSIVolumeSourceAlphaFields(podSpec, oldPodSpec)

	if !utilfeature.DefaultFeatureGate.Enabled(features.IdentifyPodOS) && !podOSInUse(oldPodSpec) {
		podSpec.OS = nil
	}

	dropDisabledTopologySpreadConstraintsFields(podSpec, oldPodSpec)
	dropDisabledNodeInclusionPolicyFields(podSpec, oldPodSpec)
}

// dropDisabledTopologySpreadConstraintsFields removes disabled fields from PodSpec related
// to TopologySpreadConstraints only if it is not already used by the old spec.
func dropDisabledTopologySpreadConstraintsFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.MinDomainsInPodTopologySpread) &&
		!minDomainsInUse(oldPodSpec) &&
		podSpec != nil {
		for i := range podSpec.TopologySpreadConstraints {
			podSpec.TopologySpreadConstraints[i].MinDomains = nil
		}
	}
}

// minDomainsInUse returns true if the pod spec is non-nil
// and has non-nil MinDomains field in TopologySpreadConstraints.
func minDomainsInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	for _, c := range podSpec.TopologySpreadConstraints {
		if c.MinDomains != nil {
			return true
		}
	}
	return false
}

// podOSInUse returns true if the pod spec is non-nil and has OS field set
func podOSInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	if podSpec.OS != nil {
		return true
	}
	return false
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

// dropDisabledCSIVolumeSourceAlphaFields removes disabled alpha fields from []CSIVolumeSource.
// This should be called from PrepareForCreate/PrepareForUpdate for all pod specs resources containing a CSIVolumeSource
func dropDisabledCSIVolumeSourceAlphaFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) && !csiInUse(oldPodSpec) {
		for i := range podSpec.Volumes {
			podSpec.Volumes[i].CSI = nil
		}
	}
}

// dropDisabledNodeInclusionPolicyFields removes disabled fields from PodSpec related
// to NodeInclusionPolicy only if it is not used by the old spec.
func dropDisabledNodeInclusionPolicyFields(podSpec, oldPodSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.NodeInclusionPolicyInPodTopologySpread) && podSpec != nil {
		if !nodeTaintsPolicyInUse(oldPodSpec) {
			for i := range podSpec.TopologySpreadConstraints {
				podSpec.TopologySpreadConstraints[i].NodeTaintsPolicy = nil
			}
		}
		if !nodeAffinityPolicyInUse(oldPodSpec) {
			for i := range podSpec.TopologySpreadConstraints {
				podSpec.TopologySpreadConstraints[i].NodeAffinityPolicy = nil
			}
		}
	}
}

// nodeAffinityPolicyInUse returns true if the pod spec is non-nil and has NodeAffinityPolicy field set
// in TopologySpreadConstraints
func nodeAffinityPolicyInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for _, c := range podSpec.TopologySpreadConstraints {
		if c.NodeAffinityPolicy != nil {
			return true
		}
	}
	return false
}

// nodeTaintsPolicyInUse returns true if the pod spec is non-nil and has NodeTaintsPolicy field set
// in TopologySpreadConstraints
func nodeTaintsPolicyInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for _, c := range podSpec.TopologySpreadConstraints {
		if c.NodeTaintsPolicy != nil {
			return true
		}
	}
	return false
}

func ephemeralContainersInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	return len(podSpec.EphemeralContainers) > 0
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

// probeGracePeriodInUse returns true if the pod spec is non-nil and has a probe that makes use
// of the probe-level terminationGracePeriodSeconds feature
func probeGracePeriodInUse(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	var inUse bool
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		// cannot be set for readiness probes
		if (c.LivenessProbe != nil && c.LivenessProbe.TerminationGracePeriodSeconds != nil) ||
			(c.StartupProbe != nil && c.StartupProbe.TerminationGracePeriodSeconds != nil) {
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

// SeccompAnnotationForField takes a pod seccomp profile field and returns the
// converted annotation value
func SeccompAnnotationForField(field *api.SeccompProfile) string {
	// If only seccomp fields are specified, add the corresponding annotations.
	// This ensures that the fields are enforced even if the node version
	// trails the API version
	switch field.Type {
	case api.SeccompProfileTypeUnconfined:
		return v1.SeccompProfileNameUnconfined

	case api.SeccompProfileTypeRuntimeDefault:
		return v1.SeccompProfileRuntimeDefault

	case api.SeccompProfileTypeLocalhost:
		if field.LocalhostProfile != nil {
			return v1.SeccompLocalhostProfileNamePrefix + *field.LocalhostProfile
		}
	}

	// we can only reach this code path if the LocalhostProfile is nil but the
	// provided field type is SeccompProfileTypeLocalhost or if an unrecognized
	// type is specified
	return ""
}

// SeccompFieldForAnnotation takes a pod annotation and returns the converted
// seccomp profile field.
func SeccompFieldForAnnotation(annotation string) *api.SeccompProfile {
	// If only seccomp annotations are specified, copy the values into the
	// corresponding fields. This ensures that existing applications continue
	// to enforce seccomp, and prevents the kubelet from needing to resolve
	// annotations & fields.
	if annotation == v1.SeccompProfileNameUnconfined {
		return &api.SeccompProfile{Type: api.SeccompProfileTypeUnconfined}
	}

	if annotation == api.SeccompProfileRuntimeDefault || annotation == api.DeprecatedSeccompProfileDockerDefault {
		return &api.SeccompProfile{Type: api.SeccompProfileTypeRuntimeDefault}
	}

	if strings.HasPrefix(annotation, v1.SeccompLocalhostProfileNamePrefix) {
		localhostProfile := strings.TrimPrefix(annotation, v1.SeccompLocalhostProfileNamePrefix)
		if localhostProfile != "" {
			return &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &localhostProfile,
			}
		}
	}

	// we can only reach this code path if the localhostProfile name has a zero
	// length or if the annotation has an unrecognized value
	return nil
}

// setsWindowsHostProcess returns true if WindowsOptions.HostProcess is set (true or false)
// anywhere in the pod spec.
func setsWindowsHostProcess(podSpec *api.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	// Check Pod's WindowsOptions.HostProcess
	if podSpec.SecurityContext != nil && podSpec.SecurityContext.WindowsOptions != nil && podSpec.SecurityContext.WindowsOptions.HostProcess != nil {
		return true
	}

	// Check WindowsOptions.HostProcess for each container
	inUse := false
	VisitContainers(podSpec, AllContainers, func(c *api.Container, containerType ContainerType) bool {
		if c.SecurityContext != nil && c.SecurityContext.WindowsOptions != nil && c.SecurityContext.WindowsOptions.HostProcess != nil {
			inUse = true
			return false
		}
		return true
	})

	return inUse
}
