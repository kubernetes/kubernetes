/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"math"
	"net"
	"reflect"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	podshelper "k8s.io/kubernetes/pkg/apis/core/pods"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

// ValidatePodName can be used to check whether the given pod name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidatePodName = NameIsDNSSubdomain

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *core.Pod) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName, fldPath)
	allErrs = append(allErrs, ValidatePodSpecificAnnotations(pod.ObjectMeta.Annotations, &pod.Spec, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec, field.NewPath("spec"))...)

	// we do additional validation only pertinent for pods and not pod templates
	// this was done to preserve backwards compatibility
	specPath := field.NewPath("spec")

	allErrs = append(allErrs, validateContainersOnlyForPod(pod.Spec.Containers, specPath.Child("containers"))...)
	allErrs = append(allErrs, validateContainersOnlyForPod(pod.Spec.InitContainers, specPath.Child("initContainers"))...)

	if utilfeature.DefaultFeatureGate.Enabled(features.HugePages) {
		hugePageResources := sets.NewString()
		for i := range pod.Spec.Containers {
			resourceSet := toContainerResourcesSet(&pod.Spec.Containers[i])
			for resourceStr := range resourceSet {
				if v1helper.IsHugePageResourceName(v1.ResourceName(resourceStr)) {
					hugePageResources.Insert(resourceStr)
				}
			}
		}
		if len(hugePageResources) > 1 {
			allErrs = append(allErrs, field.Invalid(specPath, hugePageResources, "must use a single hugepage size in a pod spec"))
		}
	}

	return allErrs
}

// ValidatePodUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodUpdate(newPod, oldPod *core.Pod) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta, fldPath)
	allErrs = append(allErrs, ValidatePodSpecificAnnotationUpdates(newPod, oldPod, fldPath.Child("annotations"))...)
	specPath := field.NewPath("spec")

	// validate updateable fields:
	// 1.  spec.containers[*].image
	// 2.  spec.initContainers[*].image
	// 3.  spec.activeDeadlineSeconds

	containerErrs, stop := ValidateContainerUpdates(newPod.Spec.Containers, oldPod.Spec.Containers, specPath.Child("containers"))
	allErrs = append(allErrs, containerErrs...)
	if stop {
		return allErrs
	}
	containerErrs, stop = ValidateContainerUpdates(newPod.Spec.InitContainers, oldPod.Spec.InitContainers, specPath.Child("initContainers"))
	allErrs = append(allErrs, containerErrs...)
	if stop {
		return allErrs
	}

	// validate updated spec.activeDeadlineSeconds.  two types of updates are allowed:
	// 1.  from nil to a positive value
	// 2.  from a positive value to a lesser, non-negative value
	if newPod.Spec.ActiveDeadlineSeconds != nil {
		newActiveDeadlineSeconds := *newPod.Spec.ActiveDeadlineSeconds
		if newActiveDeadlineSeconds < 0 || newActiveDeadlineSeconds > math.MaxInt32 {
			allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newActiveDeadlineSeconds, validation.InclusiveRangeError(0, math.MaxInt32)))
			return allErrs
		}
		if oldPod.Spec.ActiveDeadlineSeconds != nil {
			oldActiveDeadlineSeconds := *oldPod.Spec.ActiveDeadlineSeconds
			if oldActiveDeadlineSeconds < newActiveDeadlineSeconds {
				allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newActiveDeadlineSeconds, "must be less than or equal to previous value"))
				return allErrs
			}
		}
	} else if oldPod.Spec.ActiveDeadlineSeconds != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newPod.Spec.ActiveDeadlineSeconds, "must not update from a positive integer to nil value"))
	}

	// handle updateable fields by munging those fields prior to deep equal comparison.
	mungedPod := *newPod
	// munge spec.containers[*].image
	var newContainers []core.Container
	for ix, container := range mungedPod.Spec.Containers {
		container.Image = oldPod.Spec.Containers[ix].Image
		newContainers = append(newContainers, container)
	}
	mungedPod.Spec.Containers = newContainers
	// munge spec.initContainers[*].image
	var newInitContainers []core.Container
	for ix, container := range mungedPod.Spec.InitContainers {
		container.Image = oldPod.Spec.InitContainers[ix].Image
		newInitContainers = append(newInitContainers, container)
	}
	mungedPod.Spec.InitContainers = newInitContainers
	// munge spec.activeDeadlineSeconds
	mungedPod.Spec.ActiveDeadlineSeconds = nil
	if oldPod.Spec.ActiveDeadlineSeconds != nil {
		activeDeadlineSeconds := *oldPod.Spec.ActiveDeadlineSeconds
		mungedPod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	}

	// Allow only additions to tolerations updates.
	mungedPod.Spec.Tolerations = oldPod.Spec.Tolerations
	allErrs = append(allErrs, validateOnlyAddedTolerations(newPod.Spec.Tolerations, oldPod.Spec.Tolerations, specPath.Child("tolerations"))...)

	if !apiequality.Semantic.DeepEqual(mungedPod.Spec, oldPod.Spec) {
		// This diff isn't perfect, but it's a helluva lot better an "I'm not going to tell you what the difference is".
		//TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		specDiff := diff.ObjectDiff(mungedPod.Spec, oldPod.Spec)
		allErrs = append(allErrs, field.Forbidden(specPath, fmt.Sprintf("pod updates may not change fields other than `spec.containers[*].image`, `spec.initContainers[*].image`, `spec.activeDeadlineSeconds` or `spec.tolerations` (only additions to existing tolerations)\n%v", specDiff)))
	}

	return allErrs
}

// ValidatePodSpec tests that the specified PodSpec has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidatePodSpec(spec *core.PodSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	vols, vErrs := ValidateVolumes(spec.Volumes, fldPath.Child("volumes"))
	allErrs = append(allErrs, vErrs...)
	allErrs = append(allErrs, validateContainers(spec.Containers, vols, fldPath.Child("containers"))...)
	allErrs = append(allErrs, validateInitContainers(spec.InitContainers, spec.Containers, vols, fldPath.Child("initContainers"))...)
	allErrs = append(allErrs, validateRestartPolicy(&spec.RestartPolicy, fldPath.Child("restartPolicy"))...)
	allErrs = append(allErrs, validateDNSPolicy(&spec.DNSPolicy, fldPath.Child("dnsPolicy"))...)
	allErrs = append(allErrs, unversionedvalidation.ValidateLabels(spec.NodeSelector, fldPath.Child("nodeSelector"))...)
	allErrs = append(allErrs, ValidatePodSecurityContext(spec.SecurityContext, spec, fldPath, fldPath.Child("securityContext"))...)
	allErrs = append(allErrs, validateImagePullSecrets(spec.ImagePullSecrets, fldPath.Child("imagePullSecrets"))...)
	allErrs = append(allErrs, validateAffinity(spec.Affinity, fldPath.Child("affinity"))...)
	allErrs = append(allErrs, validatePodDNSConfig(spec.DNSConfig, &spec.DNSPolicy, fldPath.Child("dnsConfig"))...)
	if len(spec.ServiceAccountName) > 0 {
		for _, msg := range ValidateServiceAccountName(spec.ServiceAccountName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("serviceAccountName"), spec.ServiceAccountName, msg))
		}
	}

	if len(spec.NodeName) > 0 {
		for _, msg := range ValidateNodeName(spec.NodeName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nodeName"), spec.NodeName, msg))
		}
	}

	if spec.ActiveDeadlineSeconds != nil {
		value := *spec.ActiveDeadlineSeconds
		if value < 1 || value > math.MaxInt32 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("activeDeadlineSeconds"), value, validation.InclusiveRangeError(1, math.MaxInt32)))
		}
	}

	if len(spec.Hostname) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(spec.Hostname, fldPath.Child("hostname"))...)
	}

	if len(spec.Subdomain) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(spec.Subdomain, fldPath.Child("subdomain"))...)
	}

	if len(spec.Tolerations) > 0 {
		allErrs = append(allErrs, ValidateTolerations(spec.Tolerations, fldPath.Child("tolerations"))...)
	}

	if len(spec.HostAliases) > 0 {
		allErrs = append(allErrs, ValidateHostAliases(spec.HostAliases, fldPath.Child("hostAliases"))...)
	}

	if len(spec.PriorityClassName) > 0 {
		if utilfeature.DefaultFeatureGate.Enabled(features.PodPriority) {
			for _, msg := range ValidatePriorityClassName(spec.PriorityClassName, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("priorityClassName"), spec.PriorityClassName, msg))
			}
		}
	}

	return allErrs
}

func ValidatePodSpecificAnnotations(annotations map[string]string, spec *core.PodSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if value, isMirror := annotations[core.MirrorPodAnnotationKey]; isMirror {
		if len(spec.NodeName) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Key(core.MirrorPodAnnotationKey), value, "must set spec.nodeName if mirror pod annotation is set"))
		}
	}

	if annotations[core.TolerationsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateTolerationsInPodAnnotations(annotations, fldPath)...)
	}

	allErrs = append(allErrs, ValidateSeccompPodAnnotations(annotations, fldPath)...)
	allErrs = append(allErrs, ValidateAppArmorPodAnnotations(annotations, spec, fldPath)...)

	sysctls, err := helper.SysctlsFromPodAnnotation(annotations[core.SysctlsPodAnnotationKey])
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Key(core.SysctlsPodAnnotationKey), annotations[core.SysctlsPodAnnotationKey], err.Error()))
	} else {
		allErrs = append(allErrs, validateSysctls(sysctls, fldPath.Key(core.SysctlsPodAnnotationKey))...)
	}
	unsafeSysctls, err := helper.SysctlsFromPodAnnotation(annotations[core.UnsafeSysctlsPodAnnotationKey])
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Key(core.UnsafeSysctlsPodAnnotationKey), annotations[core.UnsafeSysctlsPodAnnotationKey], err.Error()))
	} else {
		allErrs = append(allErrs, validateSysctls(unsafeSysctls, fldPath.Key(core.UnsafeSysctlsPodAnnotationKey))...)
	}
	inBoth := sysctlIntersection(sysctls, unsafeSysctls)
	if len(inBoth) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Key(core.UnsafeSysctlsPodAnnotationKey), strings.Join(inBoth, ", "), "can not be safe and unsafe"))
	}

	return allErrs
}

// ValidateTolerationsInPodAnnotations tests that the serialized tolerations in Pod.Annotations has valid data
func ValidateTolerationsInPodAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	tolerations, err := helper.GetTolerationsFromPodAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, core.TolerationsAnnotationKey, err.Error()))
		return allErrs
	}

	if len(tolerations) > 0 {
		allErrs = append(allErrs, ValidateTolerations(tolerations, fldPath.Child(core.TolerationsAnnotationKey))...)
	}

	return allErrs
}

// ValidateTolerations tests if given tolerations have valid data.
func ValidateTolerations(tolerations []core.Toleration, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	for i, toleration := range tolerations {
		idxPath := fldPath.Index(i)
		// validate the toleration key
		if len(toleration.Key) > 0 {
			allErrors = append(allErrors, unversionedvalidation.ValidateLabelName(toleration.Key, idxPath.Child("key"))...)
		}

		// empty toleration key with Exists operator and empty value means match all taints
		if len(toleration.Key) == 0 && toleration.Operator != core.TolerationOpExists {
			allErrors = append(allErrors, field.Invalid(idxPath.Child("operator"), toleration.Operator,
				"operator must be Exists when `key` is empty, which means \"match all values and all keys\""))
		}

		if toleration.TolerationSeconds != nil && toleration.Effect != core.TaintEffectNoExecute {
			allErrors = append(allErrors, field.Invalid(idxPath.Child("effect"), toleration.Effect,
				"effect must be 'NoExecute' when `tolerationSeconds` is set"))
		}

		// validate toleration operator and value
		switch toleration.Operator {
		// empty operator means Equal
		case core.TolerationOpEqual, "":
			if errs := validation.IsValidLabelValue(toleration.Value); len(errs) != 0 {
				allErrors = append(allErrors, field.Invalid(idxPath.Child("operator"), toleration.Value, strings.Join(errs, ";")))
			}
		case core.TolerationOpExists:
			if len(toleration.Value) > 0 {
				allErrors = append(allErrors, field.Invalid(idxPath.Child("operator"), toleration, "value must be empty when `operator` is 'Exists'"))
			}
		default:
			validValues := []string{string(core.TolerationOpEqual), string(core.TolerationOpExists)}
			allErrors = append(allErrors, field.NotSupported(idxPath.Child("operator"), toleration.Operator, validValues))
		}

		// validate toleration effect, empty toleration effect means match all taint effects
		if len(toleration.Effect) > 0 {
			allErrors = append(allErrors, validateTaintEffect(&toleration.Effect, true, idxPath.Child("effect"))...)
		}
	}
	return allErrors
}

func ValidatePodSpecificAnnotationUpdates(newPod, oldPod *core.Pod, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	newAnnotations := newPod.Annotations
	oldAnnotations := oldPod.Annotations
	for k, oldVal := range oldAnnotations {
		if newVal, exists := newAnnotations[k]; exists && newVal == oldVal {
			continue // No change.
		}
		if strings.HasPrefix(k, apparmor.ContainerAnnotationKeyPrefix) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(k), "may not remove or update AppArmor annotations"))
		}
		if k == core.MirrorPodAnnotationKey {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(k), "may not remove or update mirror pod annotation"))
		}
	}
	// Check for additions
	for k := range newAnnotations {
		if _, ok := oldAnnotations[k]; ok {
			continue // No change.
		}
		if strings.HasPrefix(k, apparmor.ContainerAnnotationKeyPrefix) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(k), "may not add AppArmor annotations"))
		}
		if k == core.MirrorPodAnnotationKey {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(k), "may not add mirror pod annotation"))
		}
	}
	allErrs = append(allErrs, ValidatePodSpecificAnnotations(newAnnotations, &newPod.Spec, fldPath)...)
	return allErrs
}

// ValidateEnv validates env vars
func ValidateEnv(vars []core.EnvVar, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, ev := range vars {
		idxPath := fldPath.Index(i)
		if len(ev.Name) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("name"), ""))
		} else {
			for _, msg := range validation.IsEnvVarName(ev.Name) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), ev.Name, msg))
			}
		}
		allErrs = append(allErrs, validateEnvVarValueFrom(ev, idxPath.Child("valueFrom"))...)
	}
	return allErrs
}

func ValidateEnvFrom(vars []core.EnvFromSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, ev := range vars {
		idxPath := fldPath.Index(i)
		if len(ev.Prefix) > 0 {
			for _, msg := range validation.IsEnvVarName(ev.Prefix) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("prefix"), ev.Prefix, msg))
			}
		}

		numSources := 0
		if ev.ConfigMapRef != nil {
			numSources++
			allErrs = append(allErrs, validateConfigMapEnvSource(ev.ConfigMapRef, idxPath.Child("configMapRef"))...)
		}
		if ev.SecretRef != nil {
			numSources++
			allErrs = append(allErrs, validateSecretEnvSource(ev.SecretRef, idxPath.Child("secretRef"))...)
		}

		if numSources == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "must specify one of: `configMapRef` or `secretRef`"))
		} else if numSources > 1 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "may not have more than one field specified at a time"))
		}
	}
	return allErrs
}

// ValidatePodBinding tests if required fields in the pod binding are legal.
func ValidatePodBinding(binding *core.Binding) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(binding.Target.Kind) != 0 && binding.Target.Kind != "Node" {
		// TODO: When validation becomes versioned, this gets more complicated.
		allErrs = append(allErrs, field.NotSupported(field.NewPath("target", "kind"), binding.Target.Kind, []string{"Node", "<empty>"}))
	}
	if len(binding.Target.Name) == 0 {
		// TODO: When validation becomes versioned, this gets more complicated.
		allErrs = append(allErrs, field.Required(field.NewPath("target", "name"), ""))
	}

	return allErrs
}

func ValidatePodLogOptions(opts *core.PodLogOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if opts.TailLines != nil && *opts.TailLines < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("tailLines"), *opts.TailLines, isNegativeErrorMsg))
	}
	if opts.LimitBytes != nil && *opts.LimitBytes < 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("limitBytes"), *opts.LimitBytes, "must be greater than 0"))
	}
	switch {
	case opts.SinceSeconds != nil && opts.SinceTime != nil:
		allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "at most one of `sinceTime` or `sinceSeconds` may be specified"))
	case opts.SinceSeconds != nil:
		if *opts.SinceSeconds < 1 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("sinceSeconds"), *opts.SinceSeconds, "must be greater than 0"))
		}
	}
	return allErrs
}

// ValidatePodStatusUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodStatusUpdate(newPod, oldPod *core.Pod) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta, fldPath)
	allErrs = append(allErrs, ValidatePodSpecificAnnotationUpdates(newPod, oldPod, fldPath.Child("annotations"))...)

	fldPath = field.NewPath("status")
	if newPod.Spec.NodeName != oldPod.Spec.NodeName {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("nodeName"), "may not be changed directly"))
	}

	if newPod.Status.NominatedNodeName != oldPod.Status.NominatedNodeName && len(newPod.Status.NominatedNodeName) > 0 {
		for _, msg := range ValidateNodeName(newPod.Status.NominatedNodeName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nominatedNodeName"), newPod.Status.NominatedNodeName, msg))
		}
	}

	// If pod should not restart, make sure the status update does not transition
	// any terminated containers to a non-terminated state.
	allErrs = append(allErrs, ValidateContainerStateTransition(newPod.Status.ContainerStatuses, oldPod.Status.ContainerStatuses, fldPath.Child("containerStatuses"), oldPod.Spec.RestartPolicy)...)
	allErrs = append(allErrs, ValidateContainerStateTransition(newPod.Status.InitContainerStatuses, oldPod.Status.InitContainerStatuses, fldPath.Child("initContainerStatuses"), oldPod.Spec.RestartPolicy)...)

	// For status update we ignore changes to pod spec.
	newPod.Spec = oldPod.Spec

	return allErrs
}

// ValidateContainerStateTransition test to if any illegal container state transitions are being attempted
func ValidateContainerStateTransition(newStatuses, oldStatuses []core.ContainerStatus, fldpath *field.Path, restartPolicy core.RestartPolicy) field.ErrorList {
	allErrs := field.ErrorList{}
	// If we should always restart, containers are allowed to leave the terminated state
	if restartPolicy == core.RestartPolicyAlways {
		return allErrs
	}
	for i, oldStatus := range oldStatuses {
		// Skip any container that is not terminated
		if oldStatus.State.Terminated == nil {
			continue
		}
		// Skip any container that failed but is allowed to restart
		if oldStatus.State.Terminated.ExitCode != 0 && restartPolicy == core.RestartPolicyOnFailure {
			continue
		}
		for _, newStatus := range newStatuses {
			if oldStatus.Name == newStatus.Name && newStatus.State.Terminated == nil {
				allErrs = append(allErrs, field.Forbidden(fldpath.Index(i).Child("state"), "may not be transitioned to non-terminated state"))
			}
		}
	}
	return allErrs
}

func ValidateContainerUpdates(newContainers, oldContainers []core.Container, fldPath *field.Path) (allErrs field.ErrorList, stop bool) {
	allErrs = field.ErrorList{}
	if len(newContainers) != len(oldContainers) {
		//TODO: Pinpoint the specific container that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, field.Forbidden(fldPath, "pod updates may not add or remove containers"))
		return allErrs, true
	}

	// validate updated container images
	for i, ctr := range newContainers {
		if len(ctr.Image) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("image"), ""))
		}
		// this is only called from ValidatePodUpdate so its safe to check leading/trailing whitespace.
		if len(strings.TrimSpace(ctr.Image)) != len(ctr.Image) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("image"), ctr.Image, "must not have leading or trailing whitespace"))
		}
	}
	return allErrs, false
}

func ValidateSeccompProfile(p string, fldPath *field.Path) field.ErrorList {
	if p == "docker/default" {
		return nil
	}
	if p == "unconfined" {
		return nil
	}
	if strings.HasPrefix(p, "localhost/") {
		return validateLocalDescendingPath(strings.TrimPrefix(p, "localhost/"), fldPath)
	}
	return field.ErrorList{field.Invalid(fldPath, p, "must be a valid seccomp profile")}
}

func ValidateSeccompPodAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if p, exists := annotations[core.SeccompPodAnnotationKey]; exists {
		allErrs = append(allErrs, ValidateSeccompProfile(p, fldPath.Child(core.SeccompPodAnnotationKey))...)
	}
	for k, p := range annotations {
		if strings.HasPrefix(k, core.SeccompContainerAnnotationKeyPrefix) {
			allErrs = append(allErrs, ValidateSeccompProfile(p, fldPath.Child(k))...)
		}
	}

	return allErrs
}

func ValidateAppArmorPodAnnotations(annotations map[string]string, spec *core.PodSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for k, p := range annotations {
		if !strings.HasPrefix(k, apparmor.ContainerAnnotationKeyPrefix) {
			continue
		}
		// TODO: this belongs to admission, not general pod validation:
		if !utilfeature.DefaultFeatureGate.Enabled(features.AppArmor) {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(k), "AppArmor is disabled by feature-gate"))
			continue
		}
		containerName := strings.TrimPrefix(k, apparmor.ContainerAnnotationKeyPrefix)
		if !podSpecHasContainer(spec, containerName) {
			allErrs = append(allErrs, field.Invalid(fldPath.Key(k), containerName, "container not found"))
		}

		if err := apparmor.ValidateProfileFormat(p); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Key(k), p, err.Error()))
		}
	}

	return allErrs
}

func ValidateHostAliases(hostAliases []core.HostAlias, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, hostAlias := range hostAliases {
		if ip := net.ParseIP(hostAlias.IP); ip == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("ip"), hostAlias.IP, "must be valid IP address"))
		}
		for _, hostname := range hostAlias.Hostnames {
			allErrs = append(allErrs, ValidateDNS1123Subdomain(hostname, fldPath.Child("hostnames"))...)
		}
	}
	return allErrs
}

// ValidateAvoidPodsInNodeAnnotations tests that the serialized AvoidPods in Node.Annotations has valid data
func ValidateAvoidPodsInNodeAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	v1Avoids, err := v1helper.GetAvoidPodsFromNodeAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("AvoidPods"), core.PreferAvoidPodsAnnotationKey, err.Error()))
		return allErrs
	}
	var avoids core.AvoidPods
	if err := corev1.Convert_v1_AvoidPods_To_core_AvoidPods(&v1Avoids, &avoids, nil); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("AvoidPods"), core.PreferAvoidPodsAnnotationKey, err.Error()))
		return allErrs
	}

	if len(avoids.PreferAvoidPods) != 0 {
		for i, pa := range avoids.PreferAvoidPods {
			idxPath := fldPath.Child(core.PreferAvoidPodsAnnotationKey).Index(i)
			allErrs = append(allErrs, validatePreferAvoidPodsEntry(pa, idxPath)...)
		}
	}

	return allErrs
}

// AccumulateUniqueHostPorts extracts each HostPort of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniqueHostPorts(containers []core.Container, accumulator *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for ci, ctr := range containers {
		idxPath := fldPath.Index(ci)
		portsPath := idxPath.Child("ports")
		for pi := range ctr.Ports {
			idxPath := portsPath.Index(pi)
			port := ctr.Ports[pi].HostPort
			if port == 0 {
				continue
			}
			str := fmt.Sprintf("%s/%s/%d", ctr.Ports[pi].Protocol, ctr.Ports[pi].HostIP, port)
			if accumulator.Has(str) {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("hostPort"), str))
			} else {
				accumulator.Insert(str)
			}
		}
	}
	return allErrs
}

func podSpecHasContainer(spec *core.PodSpec, containerName string) bool {
	for _, c := range spec.InitContainers {
		if c.Name == containerName {
			return true
		}
	}
	for _, c := range spec.Containers {
		if c.Name == containerName {
			return true
		}
	}
	return false
}

// validateContainersOnlyForPod does additional validation for containers on a pod versus a pod template
// it only does additive validation of fields not covered in validateContainers
func validateContainersOnlyForPod(containers []core.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, ctr := range containers {
		idxPath := fldPath.Index(i)
		if len(ctr.Image) != len(strings.TrimSpace(ctr.Image)) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("image"), ctr.Image, "must not have leading or trailing whitespace"))
		}
	}
	return allErrs
}

func validateInitContainers(containers, otherContainers []core.Container, deviceVolumes map[string]core.VolumeSource, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(containers) > 0 {
		allErrs = append(allErrs, validateContainers(containers, deviceVolumes, fldPath)...)
	}

	allNames := sets.String{}
	for _, ctr := range otherContainers {
		allNames.Insert(ctr.Name)
	}
	for i, ctr := range containers {
		idxPath := fldPath.Index(i)
		if allNames.Has(ctr.Name) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), ctr.Name))
		}
		if len(ctr.Name) > 0 {
			allNames.Insert(ctr.Name)
		}
		if ctr.Lifecycle != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("lifecycle"), ctr.Lifecycle, "must not be set for init containers"))
		}
		if ctr.LivenessProbe != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("livenessProbe"), ctr.LivenessProbe, "must not be set for init containers"))
		}
		if ctr.ReadinessProbe != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("readinessProbe"), ctr.ReadinessProbe, "must not be set for init containers"))
		}
	}
	return allErrs
}

func validateContainers(containers []core.Container, volumes map[string]core.VolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(containers) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}

	allNames := sets.String{}
	for i, ctr := range containers {
		idxPath := fldPath.Index(i)
		namePath := idxPath.Child("name")
		volMounts := GetVolumeMountMap(ctr.VolumeMounts)
		volDevices := GetVolumeDeviceMap(ctr.VolumeDevices)

		if len(ctr.Name) == 0 {
			allErrs = append(allErrs, field.Required(namePath, ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(ctr.Name, namePath)...)
		}
		if allNames.Has(ctr.Name) {
			allErrs = append(allErrs, field.Duplicate(namePath, ctr.Name))
		} else {
			allNames.Insert(ctr.Name)
		}
		// TODO: do not validate leading and trailing whitespace to preserve backward compatibility.
		// for example: https://github.com/openshift/origin/issues/14659 image = " " is special token in pod template
		// others may have done similar
		if len(ctr.Image) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("image"), ""))
		}
		if ctr.Lifecycle != nil {
			allErrs = append(allErrs, validateLifecycle(ctr.Lifecycle, idxPath.Child("lifecycle"))...)
		}
		allErrs = append(allErrs, validateProbe(ctr.LivenessProbe, idxPath.Child("livenessProbe"))...)
		// Liveness-specific validation
		if ctr.LivenessProbe != nil && ctr.LivenessProbe.SuccessThreshold != 1 {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("livenessProbe", "successThreshold"), ctr.LivenessProbe.SuccessThreshold, "must be 1"))
		}

		switch ctr.TerminationMessagePolicy {
		case core.TerminationMessageReadFile, core.TerminationMessageFallbackToLogsOnError:
		case "":
			allErrs = append(allErrs, field.Required(idxPath.Child("terminationMessagePolicy"), "must be 'File' or 'FallbackToLogsOnError'"))
		default:
			allErrs = append(allErrs, field.Invalid(idxPath.Child("terminationMessagePolicy"), ctr.TerminationMessagePolicy, "must be 'File' or 'FallbackToLogsOnError'"))
		}

		allErrs = append(allErrs, validateProbe(ctr.ReadinessProbe, idxPath.Child("readinessProbe"))...)
		allErrs = append(allErrs, validateContainerPorts(ctr.Ports, idxPath.Child("ports"))...)
		allErrs = append(allErrs, ValidateEnv(ctr.Env, idxPath.Child("env"))...)
		allErrs = append(allErrs, ValidateEnvFrom(ctr.EnvFrom, idxPath.Child("envFrom"))...)
		allErrs = append(allErrs, ValidateVolumeMounts(ctr.VolumeMounts, volDevices, volumes, &ctr, idxPath.Child("volumeMounts"))...)
		allErrs = append(allErrs, ValidateVolumeDevices(ctr.VolumeDevices, volMounts, volumes, idxPath.Child("volumeDevices"))...)
		allErrs = append(allErrs, validatePullPolicy(ctr.ImagePullPolicy, idxPath.Child("imagePullPolicy"))...)
		allErrs = append(allErrs, ValidateResourceRequirements(&ctr.Resources, idxPath.Child("resources"))...)
		allErrs = append(allErrs, ValidateSecurityContext(ctr.SecurityContext, idxPath.Child("securityContext"))...)
	}
	// Check for colliding ports across all containers.
	allErrs = append(allErrs, checkHostPortConflicts(containers, fldPath)...)

	return allErrs
}

func validateRestartPolicy(restartPolicy *core.RestartPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	switch *restartPolicy {
	case core.RestartPolicyAlways, core.RestartPolicyOnFailure, core.RestartPolicyNever:
		break
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		validValues := []string{string(core.RestartPolicyAlways), string(core.RestartPolicyOnFailure), string(core.RestartPolicyNever)}
		allErrors = append(allErrors, field.NotSupported(fldPath, *restartPolicy, validValues))
	}

	return allErrors
}

func validateDNSPolicy(dnsPolicy *core.DNSPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	switch *dnsPolicy {
	case core.DNSClusterFirstWithHostNet, core.DNSClusterFirst, core.DNSDefault:
	case core.DNSNone:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) {
			allErrors = append(allErrors, field.Invalid(fldPath, dnsPolicy, "DNSPolicy: can not use 'None', custom pod DNS is disabled by feature gate"))
		}
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		validValues := []string{string(core.DNSClusterFirstWithHostNet), string(core.DNSClusterFirst), string(core.DNSDefault)}
		if utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) {
			validValues = append(validValues, string(core.DNSNone))
		}
		allErrors = append(allErrors, field.NotSupported(fldPath, dnsPolicy, validValues))
	}
	return allErrors
}

const (
	// Limits on various DNS parameters. These are derived from
	// restrictions in Linux libc name resolution handling.
	// Max number of DNS name servers.
	MaxDNSNameservers = 3
	// Max number of domains in search path.
	MaxDNSSearchPaths = 6
	// Max number of characters in search path.
	MaxDNSSearchListChars = 256
)

func validatePodDNSConfig(dnsConfig *core.PodDNSConfig, dnsPolicy *core.DNSPolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// Validate DNSNone case. Must provide at least one DNS name server.
	if utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) && dnsPolicy != nil && *dnsPolicy == core.DNSNone {
		if dnsConfig == nil {
			return append(allErrs, field.Required(fldPath, fmt.Sprintf("must provide `dnsConfig` when `dnsPolicy` is %s", core.DNSNone)))
		}
		if len(dnsConfig.Nameservers) == 0 {
			return append(allErrs, field.Required(fldPath.Child("nameservers"), fmt.Sprintf("must provide at least one DNS nameserver when `dnsPolicy` is %s", core.DNSNone)))
		}
	}

	if dnsConfig != nil {
		if !utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) {
			return append(allErrs, field.Forbidden(fldPath, "DNSConfig: custom pod DNS is disabled by feature gate"))
		}

		// Validate nameservers.
		if len(dnsConfig.Nameservers) > MaxDNSNameservers {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nameservers"), dnsConfig.Nameservers, fmt.Sprintf("must not have more than %v nameservers", MaxDNSNameservers)))
		}
		for i, ns := range dnsConfig.Nameservers {
			if ip := net.ParseIP(ns); ip == nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("nameservers").Index(i), ns, "must be valid IP address"))
			}
		}
		// Validate searches.
		if len(dnsConfig.Searches) > MaxDNSSearchPaths {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("searches"), dnsConfig.Searches, fmt.Sprintf("must not have more than %v search paths", MaxDNSSearchPaths)))
		}
		// Include the space between search paths.
		if len(strings.Join(dnsConfig.Searches, " ")) > MaxDNSSearchListChars {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("searches"), dnsConfig.Searches, "must not have more than 256 characters (including spaces) in the search list"))
		}
		for i, search := range dnsConfig.Searches {
			allErrs = append(allErrs, ValidateDNS1123Subdomain(search, fldPath.Child("searches").Index(i))...)
		}
		// Validate options.
		for i, option := range dnsConfig.Options {
			if len(option.Name) == 0 {
				allErrs = append(allErrs, field.Required(fldPath.Child("options").Index(i), "must not be empty"))
			}
		}
	}
	return allErrs
}

var supportedPullPolicies = sets.NewString(string(core.PullAlways), string(core.PullIfNotPresent), string(core.PullNever))

func validatePullPolicy(policy core.PullPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}

	switch policy {
	case core.PullAlways, core.PullIfNotPresent, core.PullNever:
		break
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		allErrors = append(allErrors, field.NotSupported(fldPath, policy, supportedPullPolicies.List()))
	}

	return allErrors
}

func sysctlIntersection(a []core.Sysctl, b []core.Sysctl) []string {
	lookup := make(map[string]struct{}, len(a))
	result := []string{}
	for i := range a {
		lookup[a[i].Name] = struct{}{}
	}
	for i := range b {
		if _, found := lookup[b[i].Name]; found {
			result = append(result, b[i].Name)
		}
	}
	return result
}

const (
	// a sysctl segment regex, concatenated with dots to form a sysctl name
	SysctlSegmentFmt string = "[a-z0-9]([-_a-z0-9]*[a-z0-9])?"

	// a sysctl name regex
	SysctlFmt string = "(" + SysctlSegmentFmt + "\\.)*" + SysctlSegmentFmt

	// the maximal length of a sysctl name
	SysctlMaxLength int = 253
)

var sysctlRegexp = regexp.MustCompile("^" + SysctlFmt + "$")

// IsValidSysctlName checks that the given string is a valid sysctl name,
// i.e. matches SysctlFmt.
func IsValidSysctlName(name string) bool {
	if len(name) > SysctlMaxLength {
		return false
	}
	return sysctlRegexp.MatchString(name)
}

func validateSysctls(sysctls []core.Sysctl, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, s := range sysctls {
		if len(s.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("name"), ""))
		} else if !IsValidSysctlName(s.Name) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("name"), s.Name, fmt.Sprintf("must have at most %d characters and match regex %s", SysctlMaxLength, SysctlFmt)))
		}
	}
	return allErrs
}

var supportedPortProtocols = sets.NewString(string(core.ProtocolTCP), string(core.ProtocolUDP))

func validateContainerPorts(ports []core.ContainerPort, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allNames := sets.String{}
	for i, port := range ports {
		idxPath := fldPath.Index(i)
		if len(port.Name) > 0 {
			if msgs := validation.IsValidPortName(port.Name); len(msgs) != 0 {
				for i = range msgs {
					allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), port.Name, msgs[i]))
				}
			} else if allNames.Has(port.Name) {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("containerPort"), ""))
		} else {
			for _, msg := range validation.IsValidPortNum(int(port.ContainerPort)) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("containerPort"), port.ContainerPort, msg))
			}
		}
		if port.HostPort != 0 {
			for _, msg := range validation.IsValidPortNum(int(port.HostPort)) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostPort"), port.HostPort, msg))
			}
		}
		if len(port.Protocol) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("protocol"), ""))
		} else if !supportedPortProtocols.Has(string(port.Protocol)) {
			allErrs = append(allErrs, field.NotSupported(idxPath.Child("protocol"), port.Protocol, supportedPortProtocols.List()))
		}
	}
	return allErrs
}

var validEnvDownwardAPIFieldPathExpressions = sets.NewString(
	"metadata.name",
	"metadata.namespace",
	"metadata.uid",
	"spec.nodeName",
	"spec.serviceAccountName",
	"status.hostIP",
	"status.podIP")
var validContainerResourceFieldPathExpressions = sets.NewString("limits.cpu", "limits.memory", "limits.ephemeral-storage", "requests.cpu", "requests.memory", "requests.ephemeral-storage")

func validateEnvVarValueFrom(ev core.EnvVar, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if ev.ValueFrom == nil {
		return allErrs
	}

	numSources := 0

	if ev.ValueFrom.FieldRef != nil {
		numSources++
		allErrs = append(allErrs, validateObjectFieldSelector(ev.ValueFrom.FieldRef, &validEnvDownwardAPIFieldPathExpressions, fldPath.Child("fieldRef"))...)
	}
	if ev.ValueFrom.ResourceFieldRef != nil {
		numSources++
		allErrs = append(allErrs, validateContainerResourceFieldSelector(ev.ValueFrom.ResourceFieldRef, &validContainerResourceFieldPathExpressions, fldPath.Child("resourceFieldRef"), false)...)
	}
	if ev.ValueFrom.ConfigMapKeyRef != nil {
		numSources++
		allErrs = append(allErrs, validateConfigMapKeySelector(ev.ValueFrom.ConfigMapKeyRef, fldPath.Child("configMapKeyRef"))...)
	}
	if ev.ValueFrom.SecretKeyRef != nil {
		numSources++
		allErrs = append(allErrs, validateSecretKeySelector(ev.ValueFrom.SecretKeyRef, fldPath.Child("secretKeyRef"))...)
	}

	if numSources == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "must specify one of: `fieldRef`, `resourceFieldRef`, `configMapKeyRef` or `secretKeyRef`"))
	} else if len(ev.Value) != 0 {
		if numSources != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "may not be specified when `value` is not empty"))
		}
	} else if numSources > 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "may not have more than one field specified at a time"))
	}

	return allErrs
}

func validateHandler(handler *core.Handler, fldPath *field.Path) field.ErrorList {
	numHandlers := 0
	allErrors := field.ErrorList{}
	if handler.Exec != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("exec"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateExecAction(handler.Exec, fldPath.Child("exec"))...)
		}
	}
	if handler.HTTPGet != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("httpGet"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateHTTPGetAction(handler.HTTPGet, fldPath.Child("httpGet"))...)
		}
	}
	if handler.TCPSocket != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("tcpSocket"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateTCPSocketAction(handler.TCPSocket, fldPath.Child("tcpSocket"))...)
		}
	}
	if numHandlers == 0 {
		allErrors = append(allErrors, field.Required(fldPath, "must specify a handler type"))
	}
	return allErrors
}

func validateLifecycle(lifecycle *core.Lifecycle, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart, fldPath.Child("postStart"))...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop, fldPath.Child("preStop"))...)
	}
	return allErrs
}

// validateImagePullSecrets checks to make sure the pull secrets are well
// formed.  Right now, we only expect name to be set (it's the only field).  If
// this ever changes and someone decides to set those fields, we'd like to
// know.
func validateImagePullSecrets(imagePullSecrets []core.LocalObjectReference, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	for i, currPullSecret := range imagePullSecrets {
		idxPath := fldPath.Index(i)
		strippedRef := core.LocalObjectReference{Name: currPullSecret.Name}
		if !reflect.DeepEqual(strippedRef, currPullSecret) {
			allErrors = append(allErrors, field.Invalid(idxPath, currPullSecret, "only name may be set"))
		}
	}
	return allErrors
}

// validateAffinity checks if given affinities are valid
func validateAffinity(affinity *core.Affinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if affinity != nil {
		if affinity.NodeAffinity != nil {
			allErrs = append(allErrs, validateNodeAffinity(affinity.NodeAffinity, fldPath.Child("nodeAffinity"))...)
		}
		if affinity.PodAffinity != nil {
			allErrs = append(allErrs, validatePodAffinity(affinity.PodAffinity, fldPath.Child("podAffinity"))...)
		}
		if affinity.PodAntiAffinity != nil {
			allErrs = append(allErrs, validatePodAntiAffinity(affinity.PodAntiAffinity, fldPath.Child("podAntiAffinity"))...)
		}
	}

	return allErrs
}

func validateTaintEffect(effect *core.TaintEffect, allowEmpty bool, fldPath *field.Path) field.ErrorList {
	if !allowEmpty && len(*effect) == 0 {
		return field.ErrorList{field.Required(fldPath, "")}
	}

	allErrors := field.ErrorList{}
	switch *effect {
	// TODO: Replace next line with subsequent commented-out line when implement TaintEffectNoScheduleNoAdmit.
	case core.TaintEffectNoSchedule, core.TaintEffectPreferNoSchedule, core.TaintEffectNoExecute:
		// case core.TaintEffectNoSchedule, core.TaintEffectPreferNoSchedule, core.TaintEffectNoScheduleNoAdmit, core.TaintEffectNoExecute:
	default:
		validValues := []string{
			string(core.TaintEffectNoSchedule),
			string(core.TaintEffectPreferNoSchedule),
			string(core.TaintEffectNoExecute),
			// TODO: Uncomment this block when implement TaintEffectNoScheduleNoAdmit.
			// string(core.TaintEffectNoScheduleNoAdmit),
		}
		allErrors = append(allErrors, field.NotSupported(fldPath, effect, validValues))
	}
	return allErrors
}

// validateOnlyAddedTolerations validates updated pod tolerations.
func validateOnlyAddedTolerations(newTolerations []core.Toleration, oldTolerations []core.Toleration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, old := range oldTolerations {
		found := false
		old.TolerationSeconds = nil
		for _, new := range newTolerations {
			new.TolerationSeconds = nil
			if reflect.DeepEqual(old, new) {
				found = true
				break
			}
		}
		if !found {
			allErrs = append(allErrs, field.Forbidden(fldPath, "existing toleration can not be modified except its tolerationSeconds"))
			return allErrs
		}
	}

	allErrs = append(allErrs, ValidateTolerations(newTolerations, fldPath)...)
	return allErrs
}

// validatePodAffinity tests that the specified podAffinity fields have valid data
func validatePodAffinity(podAffinity *core.PodAffinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO:Uncomment below code once RequiredDuringSchedulingRequiredDuringExecution is implemented.
	// if podAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
	//	allErrs = append(allErrs, validatePodAffinityTerms(podAffinity.RequiredDuringSchedulingRequiredDuringExecution, false,
	//		fldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
	//}
	if podAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		allErrs = append(allErrs, validatePodAffinityTerms(podAffinity.RequiredDuringSchedulingIgnoredDuringExecution,
			fldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
	}
	if podAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		allErrs = append(allErrs, validateWeightedPodAffinityTerms(podAffinity.PreferredDuringSchedulingIgnoredDuringExecution,
			fldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
	}
	return allErrs
}

// validatePodAffinityTerm tests that the specified podAffinityTerm fields have valid data
func validatePodAffinityTerm(podAffinityTerm core.PodAffinityTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(podAffinityTerm.LabelSelector, fldPath.Child("matchExpressions"))...)
	for _, name := range podAffinityTerm.Namespaces {
		for _, msg := range ValidateNamespaceName(name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), name, msg))
		}
	}
	if len(podAffinityTerm.TopologyKey) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("topologyKey"), "can not be empty"))
	}
	return append(allErrs, unversionedvalidation.ValidateLabelName(podAffinityTerm.TopologyKey, fldPath.Child("topologyKey"))...)
}

// validatePodAffinityTerms tests that the specified podAffinityTerms fields have valid data
func validatePodAffinityTerms(podAffinityTerms []core.PodAffinityTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, podAffinityTerm := range podAffinityTerms {
		allErrs = append(allErrs, validatePodAffinityTerm(podAffinityTerm, fldPath.Index(i))...)
	}
	return allErrs
}

// validateWeightedPodAffinityTerms tests that the specified weightedPodAffinityTerms fields have valid data
func validateWeightedPodAffinityTerms(weightedPodAffinityTerms []core.WeightedPodAffinityTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for j, weightedTerm := range weightedPodAffinityTerms {
		if weightedTerm.Weight <= 0 || weightedTerm.Weight > 100 {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(j).Child("weight"), weightedTerm.Weight, "must be in the range 1-100"))
		}
		allErrs = append(allErrs, validatePodAffinityTerm(weightedTerm.PodAffinityTerm, fldPath.Index(j).Child("podAffinityTerm"))...)
	}
	return allErrs
}

// validatePodAntiAffinity tests that the specified podAntiAffinity fields have valid data
func validatePodAntiAffinity(podAntiAffinity *core.PodAntiAffinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO:Uncomment below code once RequiredDuringSchedulingRequiredDuringExecution is implemented.
	// if podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
	//	allErrs = append(allErrs, validatePodAffinityTerms(podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution, false,
	//		fldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
	//}
	if podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		allErrs = append(allErrs, validatePodAffinityTerms(podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution,
			fldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
	}
	if podAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		allErrs = append(allErrs, validateWeightedPodAffinityTerms(podAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution,
			fldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
	}
	return allErrs
}

// validatePreferAvoidPodsEntry tests if given PreferAvoidPodsEntry has valid data.
func validatePreferAvoidPodsEntry(avoidPodEntry core.PreferAvoidPodsEntry, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if avoidPodEntry.PodSignature.PodController == nil {
		allErrors = append(allErrors, field.Required(fldPath.Child("PodSignature"), ""))
	} else {
		if *(avoidPodEntry.PodSignature.PodController.Controller) != true {
			allErrors = append(allErrors,
				field.Invalid(fldPath.Child("PodSignature").Child("PodController").Child("Controller"),
					*(avoidPodEntry.PodSignature.PodController.Controller), "must point to a controller"))
		}
	}
	return allErrors
}

func toContainerResourcesSet(ctr *core.Container) sets.String {
	resourceNames := toResourceNames(ctr.Resources.Requests)
	resourceNames = append(resourceNames, toResourceNames(ctr.Resources.Limits)...)
	return toSet(resourceNames)
}

func toSet(resourceNames []core.ResourceName) sets.String {
	result := sets.NewString()
	for _, resourceName := range resourceNames {
		result.Insert(string(resourceName))
	}
	return result
}

func toResourceNames(resources core.ResourceList) []core.ResourceName {
	result := []core.ResourceName{}
	for resourceName := range resources {
		result = append(result, resourceName)
	}
	return result
}

func validateTCPSocketAction(tcp *core.TCPSocketAction, fldPath *field.Path) field.ErrorList {
	return ValidatePortNumOrName(tcp.Port, fldPath.Child("port"))
}

// checkHostPortConflicts checks for colliding Port.HostPort values across
// a slice of containers.
func checkHostPortConflicts(containers []core.Container, fldPath *field.Path) field.ErrorList {
	allPorts := sets.String{}
	return AccumulateUniqueHostPorts(containers, &allPorts, fldPath)
}

func validateExecAction(exec *core.ExecAction, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("command"), ""))
	}
	return allErrors
}

var supportedHTTPSchemes = sets.NewString(string(core.URISchemeHTTP), string(core.URISchemeHTTPS))

func validateHTTPGetAction(http *core.HTTPGetAction, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("path"), ""))
	}
	allErrors = append(allErrors, ValidatePortNumOrName(http.Port, fldPath.Child("port"))...)
	if !supportedHTTPSchemes.Has(string(http.Scheme)) {
		allErrors = append(allErrors, field.NotSupported(fldPath.Child("scheme"), http.Scheme, supportedHTTPSchemes.List()))
	}
	for _, header := range http.HTTPHeaders {
		for _, msg := range validation.IsHTTPHeaderName(header.Name) {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("httpHeaders"), header.Name, msg))
		}
	}
	return allErrors
}

func validateProbe(probe *core.Probe, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if probe == nil {
		return allErrs
	}
	allErrs = append(allErrs, validateHandler(&probe.Handler, fldPath)...)

	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.InitialDelaySeconds), fldPath.Child("initialDelaySeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.TimeoutSeconds), fldPath.Child("timeoutSeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.PeriodSeconds), fldPath.Child("periodSeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.SuccessThreshold), fldPath.Child("successThreshold"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.FailureThreshold), fldPath.Child("failureThreshold"))...)
	return allErrs
}

func validateSecretKeySelector(s *core.SecretKeySelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	nameFn := ValidateNameFunc(ValidateSecretName)
	for _, msg := range nameFn(s.Name, false) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), s.Name, msg))
	}
	if len(s.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	} else {
		for _, msg := range validation.IsConfigMapKey(s.Key) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("key"), s.Key, msg))
		}
	}

	return allErrs
}

func validateConfigMapKeySelector(s *core.ConfigMapKeySelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	nameFn := ValidateNameFunc(ValidateSecretName)
	for _, msg := range nameFn(s.Name, false) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), s.Name, msg))
	}
	if len(s.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	} else {
		for _, msg := range validation.IsConfigMapKey(s.Key) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("key"), s.Key, msg))
		}
	}

	return allErrs
}

func validateObjectFieldSelector(fs *core.ObjectFieldSelector, expressions *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(fs.APIVersion) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("apiVersion"), ""))
		return allErrs
	}
	if len(fs.FieldPath) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("fieldPath"), ""))
		return allErrs
	}

	internalFieldPath, _, err := podshelper.ConvertDownwardAPIFieldLabel(fs.APIVersion, fs.FieldPath, "")
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("fieldPath"), fs.FieldPath, fmt.Sprintf("error converting fieldPath: %v", err)))
		return allErrs
	}

	if path, subscript, ok := fieldpath.SplitMaybeSubscriptedPath(internalFieldPath); ok {
		switch path {
		case "metadata.annotations":
			for _, msg := range validation.IsQualifiedName(strings.ToLower(subscript)) {
				allErrs = append(allErrs, field.Invalid(fldPath, subscript, msg))
			}
		case "metadata.labels":
			for _, msg := range validation.IsQualifiedName(subscript) {
				allErrs = append(allErrs, field.Invalid(fldPath, subscript, msg))
			}
		default:
			allErrs = append(allErrs, field.Invalid(fldPath, path, "does not support subscript"))
		}
	} else if !expressions.Has(path) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("fieldPath"), path, expressions.List()))
		return allErrs
	}

	return allErrs
}

func fsResourceIsEphemeralStorage(resource string) bool {
	if resource == "limits.ephemeral-storage" || resource == "requests.ephemeral-storage" {
		return true
	}
	return false
}

func validateContainerResourceFieldSelector(fs *core.ResourceFieldSelector, expressions *sets.String, fldPath *field.Path, volume bool) field.ErrorList {
	allErrs := field.ErrorList{}

	if volume && len(fs.ContainerName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("containerName"), ""))
	} else if len(fs.Resource) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), ""))
	} else if !expressions.Has(fs.Resource) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("resource"), fs.Resource, expressions.List()))
	} else if fsResourceIsEphemeralStorage(fs.Resource) && !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "Containers' ephemeral storage requests/limits disabled by feature-gate for Downward API"))
	}
	allErrs = append(allErrs, validateContainerResourceDivisor(fs.Resource, fs.Divisor, fldPath)...)
	return allErrs
}

func validateConfigMapEnvSource(configMapSource *core.ConfigMapEnvSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(configMapSource.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else {
		for _, msg := range ValidateConfigMapName(configMapSource.Name, true) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), configMapSource.Name, msg))
		}
	}
	return allErrs
}

func validateSecretEnvSource(secretSource *core.SecretEnvSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(secretSource.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else {
		for _, msg := range ValidateSecretName(secretSource.Name, true) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), secretSource.Name, msg))
		}
	}
	return allErrs
}

var validContainerResourceDivisorForCPU = sets.NewString("1m", "1")
var validContainerResourceDivisorForMemory = sets.NewString("1", "1k", "1M", "1G", "1T", "1P", "1E", "1Ki", "1Mi", "1Gi", "1Ti", "1Pi", "1Ei")
var validContainerResourceDivisorForEphemeralStorage = sets.NewString("1", "1k", "1M", "1G", "1T", "1P", "1E", "1Ki", "1Mi", "1Gi", "1Ti", "1Pi", "1Ei")

func validateContainerResourceDivisor(rName string, divisor resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	unsetDivisor := resource.Quantity{}
	if unsetDivisor.Cmp(divisor) == 0 {
		return allErrs
	}
	switch rName {
	case "limits.cpu", "requests.cpu":
		if !validContainerResourceDivisorForCPU.Has(divisor.String()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("divisor"), rName, "only divisor's values 1m and 1 are supported with the cpu resource"))
		}
	case "limits.memory", "requests.memory":
		if !validContainerResourceDivisorForMemory.Has(divisor.String()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("divisor"), rName, "only divisor's values 1, 1k, 1M, 1G, 1T, 1P, 1E, 1Ki, 1Mi, 1Gi, 1Ti, 1Pi, 1Ei are supported with the memory resource"))
		}
	case "limits.ephemeral-storage", "requests.ephemeral-storage":
		if !validContainerResourceDivisorForEphemeralStorage.Has(divisor.String()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("divisor"), rName, "only divisor's values 1, 1k, 1M, 1G, 1T, 1P, 1E, 1Ki, 1Mi, 1Gi, 1Ti, 1Pi, 1Ei are supported with the local ephemeral storage resource"))
		}
	}
	return allErrs
}
