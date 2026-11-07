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

package validation

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metavalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
)

// PodValidationOptions contains the different settings for pod validation
type PodValidationOptions struct {
	// Allow invalid pod-deletion-cost annotation value for backward compatibility.
	AllowInvalidPodDeletionCost bool
	// Allow invalid label-value in LabelSelector
	AllowInvalidLabelValueInSelector bool
	// Allow pod spec to use non-integer multiple of huge page unit size
	AllowIndivisibleHugePagesValues bool
	// Allow invalid topologySpreadConstraint labelSelector for backward compatibility
	AllowInvalidTopologySpreadConstraintLabelSelector bool
	// Allow projected token volumes with non-local paths
	AllowNonLocalProjectedTokenPath bool
	// Allow namespaced sysctls in hostNet and hostIPC pods
	AllowNamespacedSysctlsForHostNetAndHostIPC bool
	// The top-level resource being validated is a Pod, not just a PodSpec
	// embedded in some other resource.
	ResourceIsPod bool
	// Allow relaxed validation of environment variable names
	AllowRelaxedEnvironmentVariableValidation bool
	// Allow only Recursive value of SELinuxChangePolicy.
	AllowOnlyRecursiveSELinuxChangePolicy bool
	// Indicates whether PodLevelResources feature is enabled or disabled.
	PodLevelResourcesEnabled bool
	// Indicates whether InPlacePodLevelResourcesVerticalScaling feature is enabled
	// or disabled.
	InPlacePodLevelResourcesVerticalScalingEnabled bool
	// Indicates whether InPlacePodVerticalScalingMemoryBackedVolumes feature is enabled
	// or disabled.
	InPlacePodVerticalScalingMemoryBackedVolumesEnabled bool
	// Allow invalid label-value in RequiredNodeSelector
	AllowInvalidLabelValueInRequiredNodeAffinity bool
	// Allow feature of MatchLabelKeys in TopologySpreadConstraints
	AllowMatchLabelKeysInPodTopologySpread bool
	// Allow merging selectors built from MatchLabelKeys into LabelSelector of TopologySpreadConstraints
	AllowMatchLabelKeysInPodTopologySpreadSelectorMerge bool
	// OldPod has invalid MatchLabelKeys in TopologySpreadConstraints against current(>=v1.34) validation
	OldPodViolatesMatchLabelKeysValidation bool
	// OldPod has invalid MatchLabelKeys in TopologySpreadConstraints against legacy(<v1.34) validation
	OldPodViolatesLegacyMatchLabelKeysValidation bool
	// Allows containers to consume environment variables via environment variable files.
	AllowEnvFilesValidation bool
	// Allows containers have restart policy and restart policy rules.
	AllowContainerRestartPolicyRules bool
	// Allow existing RestartContainer resize policy for non-sidecar init containers for backward compatibility
	AllowExistingRestartContainerForNonSidecarInitContainer bool
	// Allow user namespaces with volume devices, even though they will not function properly (should only be tolerated in updates of objects which already have this invalid configuration).
	AllowUserNamespacesWithVolumeDevices bool
	// Allow taint toleration comparison operators (Lt, Gt)
	AllowTaintTolerationComparisonOperators bool
	// Allow hostNetwork pods to use user namespaces
	AllowUserNamespacesHostNetworkSupport bool
	// Allow containers to have restart policy rule with RestartAllContainers action, applicable for init, sidecar, and regular containers.
	AllowRestartAllContainers bool
	// Allows container statuses to contain image volume digest
	AllowImageVolumeWithDigest bool
	// Allow empty image volume reference for backward compatibility
	AllowEmptyImageVolumeReference bool
	// Allow containers to have CAP_SYS_ADMIN even if AllowPrivilegeEscalation is false
	AllowSysAdminWhenPrivilegeEscalationFalse bool
	// Allow seccomp profiles of type OCI
	AllowSecurityProfileOCI bool
	// Allow podCertificate volumes to specify ML-DSA algorithms in the keyType field
	AllowMLDSAPodCertificateKeyTypes bool
}

func hasIndivisibleHugePagesValue(list core.ResourceList) bool {
	for resourceName, quantity := range list {
		if helper.IsHugePageResourceName(resourceName) && !helper.IsHugePageResourceValueDivisible(resourceName, quantity) {
			return true
		}
	}

	return false
}

func checkContainerUseIndivisibleHugePagesValues(container core.Container) bool {
	return hasIndivisibleHugePagesValue(container.Resources.Limits) ||
		hasIndivisibleHugePagesValue(container.Resources.Requests)
}

// usesIndivisibleHugePagesValues returns true if the pod spec asks for a non-integer
// multiple of huge page unit size anywhere validation checks for one.
func usesIndivisibleHugePagesValues(podSpec *core.PodSpec) bool {
	foundIndivisibleHugePagesValue := false
	podutil.VisitContainers(podSpec, podutil.AllContainers, func(c *core.Container, containerType podutil.ContainerType) bool {
		if checkContainerUseIndivisibleHugePagesValues(*c) {
			foundIndivisibleHugePagesValue = true
		}
		return !foundIndivisibleHugePagesValue // continue visiting if we haven't seen an invalid value yet
	})

	if foundIndivisibleHugePagesValue {
		return true
	}

	// Pod-level resources go through the same check, so an old pod carrying an
	// indivisible one has to stay updatable.
	if podSpec.Resources != nil {
		if hasIndivisibleHugePagesValue(podSpec.Resources.Limits) ||
			hasIndivisibleHugePagesValue(podSpec.Resources.Requests) {
			return true
		}
	}

	return hasIndivisibleHugePagesValue(podSpec.Overhead)
}

// hasInvalidTopologySpreadConstraintLabelSelector return true if spec.TopologySpreadConstraints have any entry with invalid labelSelector
func hasInvalidTopologySpreadConstraintLabelSelector(spec *core.PodSpec) bool {
	for _, constraint := range spec.TopologySpreadConstraints {
		errs := metavalidation.ValidateLabelSelector(constraint.LabelSelector, metavalidation.LabelSelectorValidationOptions{AllowInvalidLabelValueInSelector: false}, nil)
		if len(errs) != 0 {
			return true
		}
	}
	return false
}

// hasInvalidTopologySpreadConstrainMatchLabelKeys return true if spec.TopologySpreadConstraints have any entry with invalid MatchLabelKeys
func hasInvalidTopologySpreadConstrainMatchLabelKeys(spec *core.PodSpec) bool {
	for _, constraint := range spec.TopologySpreadConstraints {
		errs := ValidateMatchLabelKeysAndMismatchLabelKeys(nil, constraint.MatchLabelKeys, nil, constraint.LabelSelector)
		if len(errs) != 0 {
			return true
		}
	}
	return false
}

// hasLegacyInvalidTopologySpreadConstrainMatchLabelKeys return true if spec.TopologySpreadConstraints have any entry with invalid MatchLabelKeys against legacy validation
func hasLegacyInvalidTopologySpreadConstrainMatchLabelKeys(spec *core.PodSpec) bool {
	for _, constraint := range spec.TopologySpreadConstraints {
		errs := ValidateMatchLabelKeysInTopologySpread(nil, constraint.MatchLabelKeys, constraint.LabelSelector)
		if len(errs) != 0 {
			return true
		}
	}
	return false
}

// hasNonLocalProjectedTokenPath return true if spec.Volumes have any entry with non-local projected token path
func hasNonLocalProjectedTokenPath(spec *core.PodSpec) bool {
	for _, volume := range spec.Volumes {
		if volume.Projected != nil {
			for _, source := range volume.Projected.Sources {
				if source.ServiceAccountToken == nil {
					continue
				}
				errs := ValidateLocalNonReservedPath(source.ServiceAccountToken.Path, nil)
				if len(errs) != 0 {
					return true
				}
			}
		}
	}
	return false
}

// GetValidationOptionsFromPodSpecAndMeta returns validation options based on pod specs and metadata
func GetValidationOptionsFromPodSpecAndMeta(podSpec, oldPodSpec *core.PodSpec, podMeta, oldPodMeta *metav1.ObjectMeta) PodValidationOptions {
	// default pod validation options based on feature gate
	opts := PodValidationOptions{
		AllowInvalidPodDeletionCost: !utilfeature.DefaultFeatureGate.Enabled(features.PodDeletionCost),
		// Do not allow pod spec to use non-integer multiple of huge page unit size default
		AllowIndivisibleHugePagesValues:                     false,
		AllowInvalidLabelValueInSelector:                    false,
		AllowInvalidTopologySpreadConstraintLabelSelector:   false,
		AllowNamespacedSysctlsForHostNetAndHostIPC:          false,
		AllowNonLocalProjectedTokenPath:                     false,
		PodLevelResourcesEnabled:                            utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources),
		AllowInvalidLabelValueInRequiredNodeAffinity:        false,
		AllowMatchLabelKeysInPodTopologySpread:              utilfeature.DefaultFeatureGate.Enabled(features.MatchLabelKeysInPodTopologySpread),
		AllowMatchLabelKeysInPodTopologySpreadSelectorMerge: utilfeature.DefaultFeatureGate.Enabled(features.MatchLabelKeysInPodTopologySpreadSelectorMerge),
		InPlacePodLevelResourcesVerticalScalingEnabled:      utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodLevelResourcesVerticalScaling),
		InPlacePodVerticalScalingMemoryBackedVolumesEnabled: utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes),
		OldPodViolatesMatchLabelKeysValidation:              false,
		OldPodViolatesLegacyMatchLabelKeysValidation:        false,
		AllowContainerRestartPolicyRules:                    utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules),
		AllowUserNamespacesWithVolumeDevices:                false,
		// This also allows restart rules on sidecar containers.
		AllowRestartAllContainers:                               utilfeature.DefaultFeatureGate.Enabled(features.RestartAllContainersOnContainerExits),
		AllowImageVolumeWithDigest:                              utilfeature.DefaultFeatureGate.Enabled(features.ImageVolumeWithDigest),
		AllowExistingRestartContainerForNonSidecarInitContainer: hasRestartContainerForNonSidecarInitContainer(oldPodSpec),
		AllowSysAdminWhenPrivilegeEscalationFalse:               false,
		AllowMLDSAPodCertificateKeyTypes:                        utilfeature.DefaultFeatureGate.Enabled(features.PodCertificateMLDSA),
		AllowSecurityProfileOCI:                                 utilfeature.DefaultFeatureGate.Enabled(features.SecurityProfileOCI),
	}

	// If old spec uses relaxed validation or enabled the RelaxedEnvironmentVariableValidation feature gate,
	// we must allow it
	opts.AllowRelaxedEnvironmentVariableValidation = useRelaxedEnvironmentVariableValidation(podSpec, oldPodSpec)
	opts.AllowEnvFilesValidation = useAllowEnvFilesValidation(oldPodSpec)
	opts.AllowUserNamespacesHostNetworkSupport = useAllowUserNamespacesHostNetworkSupport(oldPodSpec)

	opts.AllowOnlyRecursiveSELinuxChangePolicy = useOnlyRecursiveSELinuxChangePolicy(oldPodSpec)
	opts.AllowTaintTolerationComparisonOperators = allowTaintTolerationComparisonOperators(oldPodSpec)

	if oldPodSpec != nil {
		// if old spec used non-integer multiple of huge page unit size, we must allow it
		opts.AllowIndivisibleHugePagesValues = usesIndivisibleHugePagesValues(oldPodSpec)

		opts.AllowInvalidLabelValueInSelector = hasInvalidLabelValueInAffinitySelector(oldPodSpec)
		opts.AllowInvalidLabelValueInRequiredNodeAffinity = hasInvalidLabelValueInRequiredNodeAffinity(oldPodSpec)
		// if old spec has invalid labelSelector in topologySpreadConstraint, we must allow it
		opts.AllowInvalidTopologySpreadConstraintLabelSelector = hasInvalidTopologySpreadConstraintLabelSelector(oldPodSpec)
		if opts.AllowMatchLabelKeysInPodTopologySpread {
			if opts.AllowMatchLabelKeysInPodTopologySpreadSelectorMerge {
				// If old spec has invalid MatchLabelKeys, we must set true
				opts.OldPodViolatesMatchLabelKeysValidation = hasInvalidTopologySpreadConstrainMatchLabelKeys(oldPodSpec)
			} else {
				// If old spec has invalid MatchLabelKeys against legacy validation, we must set true
				opts.OldPodViolatesLegacyMatchLabelKeysValidation = hasLegacyInvalidTopologySpreadConstrainMatchLabelKeys(oldPodSpec)
			}
		}
		// if old spec has an invalid projected token volume path, we must allow it
		opts.AllowNonLocalProjectedTokenPath = hasNonLocalProjectedTokenPath(oldPodSpec)

		// if old spec has invalid sysctl with hostNet or hostIPC, we must allow it when update
		if oldPodSpec.SecurityContext != nil && len(oldPodSpec.SecurityContext.Sysctls) != 0 {
			for _, s := range oldPodSpec.SecurityContext.Sysctls {
				err := ValidateHostSysctl(s.Name, oldPodSpec, nil)
				if err != nil {
					opts.AllowNamespacedSysctlsForHostNetAndHostIPC = true
					break
				}
			}
		}

		opts.AllowContainerRestartPolicyRules = opts.AllowContainerRestartPolicyRules || podutil.ContainerRestartRulesInUse(oldPodSpec)
		opts.AllowRestartAllContainers = opts.AllowRestartAllContainers || restartAllContainersActionInUse(oldPodSpec)

		// If old spec has userns and volume devices (doesn't work), we still allow
		// modifications to it.
		opts.AllowUserNamespacesWithVolumeDevices = hasUserNamespacesWithVolumeDevices(oldPodSpec)

		// If old spec already had an image volume with empty reference, allow it
		opts.AllowEmptyImageVolumeReference = hasEmptyImageVolumeReference(oldPodSpec)

		opts.AllowSysAdminWhenPrivilegeEscalationFalse = useAllowSysAdminWhenPrivilegeEscalationFalse(oldPodSpec)

		// If old spec has a projected pod certificate requesting an ML-DSA key type, allow it
		opts.AllowMLDSAPodCertificateKeyTypes = opts.AllowMLDSAPodCertificateKeyTypes || hasMLDSAPodCertificateProjection(oldPodSpec.Volumes)

		// If old spec uses a seccomp profile of type OCI, allow it
		opts.AllowSecurityProfileOCI = opts.AllowSecurityProfileOCI || seccompProfileOCIInUse(oldPodSpec)
	}
	if oldPodMeta != nil && !opts.AllowInvalidPodDeletionCost {
		// This is an update, so validate only if the existing object was valid.
		_, err := helper.GetDeletionCostFromPodAnnotations(oldPodMeta.Annotations)
		opts.AllowInvalidPodDeletionCost = err != nil
	}

	return opts
}

func useRelaxedEnvironmentVariableValidation(podSpec, oldPodSpec *core.PodSpec) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.RelaxedEnvironmentVariableValidation) {
		return true
	}

	var oldPodEnvVarNames, podEnvVarNames sets.Set[string]
	if oldPodSpec != nil {
		oldPodEnvVarNames = gatherPodEnvVarNames(oldPodSpec)
	}

	if podSpec != nil {
		podEnvVarNames = gatherPodEnvVarNames(podSpec)
	}

	for env := range podEnvVarNames {
		if relaxedEnvVarUsed(env, oldPodEnvVarNames) {
			return true
		}
	}

	return false
}

func useAllowUserNamespacesHostNetworkSupport(oldPodSpec *core.PodSpec) bool {
	// Return true early if feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.UserNamespacesHostNetworkSupport) {
		return true
	}

	if oldPodSpec == nil || oldPodSpec.HostUsers == nil {
		return false
	}

	// If a pod with user namespaces and hostNetwork already exists in the cluster,
	// this allows it to continue using the UserNamespacesHostNetworkSupport
	// validation logic even after the feature gate is disabled.
	userNamespaces := !*oldPodSpec.HostUsers
	return oldPodSpec.HostNetwork && userNamespaces
}

func useAllowEnvFilesValidation(oldPodSpec *core.PodSpec) bool {
	// Return true early if feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.EnvFiles) {
		return true
	}

	if oldPodSpec == nil {
		return false
	}

	for c := range podutil.ContainerIter(oldPodSpec, podutil.AllContainers) {
		if hasEnvFileKeyRef(c.Env) {
			return true
		}
	}

	return false
}

func hasEnvFileKeyRef(envs []core.EnvVar) bool {
	for _, env := range envs {
		if env.ValueFrom != nil && env.ValueFrom.FileKeyRef != nil {
			return true
		}
	}
	return false
}

func useAllowSysAdminWhenPrivilegeEscalationFalse(oldPodSpec *core.PodSpec) bool {
	if oldPodSpec == nil {
		return false
	}

	for c := range podutil.ContainerIter(oldPodSpec, podutil.AllContainers) {
		if hasSysAdminAndPrivilegeEscalationFalse(c.SecurityContext) {
			return true
		}
	}

	return false
}

func hasSysAdminAndPrivilegeEscalationFalse(sc *core.SecurityContext) bool {
	if sc == nil {
		return false
	}
	if sc.AllowPrivilegeEscalation != nil && *sc.AllowPrivilegeEscalation {
		return false
	}
	if sc.Capabilities == nil {
		return false
	}
	for _, c := range sc.Capabilities.Add {
		if string(c) == "CAP_SYS_ADMIN" {
			return true
		}
	}
	return false
}

func hasMLDSAPodCertificateProjection(volumes []core.Volume) bool {
	for _, volume := range volumes {
		if volume.Projected != nil {
			for _, source := range volume.Projected.Sources {
				if source.PodCertificate != nil && (source.PodCertificate.KeyType == "MLDSA44" || source.PodCertificate.KeyType == "MLDSA65" || source.PodCertificate.KeyType == "MLDSA87") {
					return true
				}
			}
		}
	}

	return false
}

func gatherPodEnvVarNames(podSpec *core.PodSpec) sets.Set[string] {
	podEnvVarNames := sets.Set[string]{}

	for _, c := range podSpec.Containers {
		for _, env := range c.Env {
			podEnvVarNames.Insert(env.Name)
		}

		for _, env := range c.EnvFrom {
			podEnvVarNames.Insert(env.Prefix)
		}
	}

	for _, c := range podSpec.InitContainers {
		for _, env := range c.Env {
			podEnvVarNames.Insert(env.Name)
		}

		for _, env := range c.EnvFrom {
			podEnvVarNames.Insert(env.Prefix)
		}
	}

	for _, c := range podSpec.EphemeralContainers {
		for _, env := range c.Env {
			podEnvVarNames.Insert(env.Name)
		}

		for _, env := range c.EnvFrom {
			podEnvVarNames.Insert(env.Prefix)
		}
	}

	return podEnvVarNames
}

func relaxedEnvVarUsed(name string, oldPodEnvVarNames sets.Set[string]) bool {
	// A length of 0 means this is not an update request,
	// or the old pod does not exist in the env.
	// We will let the feature gate decide whether to use relaxed rules.
	if oldPodEnvVarNames.Len() == 0 {
		return false
	}

	if len(validation.IsEnvVarName(name)) == 0 || len(validation.IsRelaxedEnvVarName(name)) != 0 {
		// It's either a valid name by strict rules or an invalid name under relaxed rules.
		// Either way, we'll use strict rules to validate.
		return false
	}

	// The name in question failed strict rules but passed relaxed rules.
	if oldPodEnvVarNames.Has(name) {
		// This relaxed-rules name was already in use.
		return true
	}

	return false
}

// GetValidationOptionsFromPodTemplate will return pod validation options for specified template.
func GetValidationOptionsFromPodTemplate(podTemplate, oldPodTemplate *core.PodTemplateSpec) PodValidationOptions {
	var newPodSpec, oldPodSpec *core.PodSpec
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

func hasInvalidLabelValueInAffinitySelector(spec *core.PodSpec) bool {
	if spec.Affinity != nil {
		if spec.Affinity.PodAffinity != nil {
			for _, term := range spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution {
				allErrs := ValidatePodAffinityTermSelector(term, false, nil)
				if len(allErrs) != 0 {
					return true
				}
			}
			for _, term := range spec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution {
				allErrs := ValidatePodAffinityTermSelector(term.PodAffinityTerm, false, nil)
				if len(allErrs) != 0 {
					return true
				}
			}
		}
		if spec.Affinity.PodAntiAffinity != nil {
			for _, term := range spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution {
				allErrs := ValidatePodAffinityTermSelector(term, false, nil)
				if len(allErrs) != 0 {
					return true
				}
			}
			for _, term := range spec.Affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution {
				allErrs := ValidatePodAffinityTermSelector(term.PodAffinityTerm, false, nil)
				if len(allErrs) != 0 {
					return true
				}
			}
		}
	}
	return false
}

func hasInvalidLabelValueInRequiredNodeAffinity(spec *core.PodSpec) bool {
	if spec == nil ||
		spec.Affinity == nil ||
		spec.Affinity.NodeAffinity == nil ||
		spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return false
	}
	return helper.HasInvalidLabelValueInNodeSelectorTerms(spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms)
}

func useOnlyRecursiveSELinuxChangePolicy(oldPodSpec *core.PodSpec) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMount) {
		// All policies are allowed
		return false
	}

	if podutil.SELinuxChangePolicyInUse(oldPodSpec) {
		// The old pod spec has *any* policy: we need to keep that object update-able.
		return false
	}
	// No feature gate + no value in the old object -> only Recursive is allowed
	return true
}

func taintTolerationComparisonOperatorsInUse(podSpec *core.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for _, toleration := range podSpec.Tolerations {
		if toleration.Operator == core.TolerationOpLt || toleration.Operator == core.TolerationOpGt {
			return true
		}
	}
	return false
}

func allowTaintTolerationComparisonOperators(oldPodSpec *core.PodSpec) bool {
	// allow the operators if the feature gate is enabled or the old pod spec uses
	// comparison operators
	if utilfeature.DefaultFeatureGate.Enabled(features.TaintTolerationComparisonOperators) ||
		taintTolerationComparisonOperatorsInUse(oldPodSpec) {
		return true
	}
	return false
}

func hasUserNamespacesWithVolumeDevices(podSpec *core.PodSpec) bool {
	if podSpec.HostUsers == nil || *podSpec.HostUsers {
		return false
	}

	hasVolumeDevices := false
	podutil.VisitContainers(podSpec, podutil.AllContainers, func(c *core.Container, _ podutil.ContainerType) bool {
		if len(c.VolumeDevices) > 0 {
			hasVolumeDevices = true
			return false // stop iterating
		}
		return true // keep iterating
	})
	return hasVolumeDevices
}

// hasEmptyImageVolumeReference returns true if the pod spec has any image volume with an empty reference.
func hasEmptyImageVolumeReference(podSpec *core.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	for _, v := range podSpec.Volumes {
		if v.Image != nil && len(v.Image.Reference) == 0 {
			return true
		}
	}
	return false
}

func restartAllContainersActionInUse(oldPodSpec *core.PodSpec) bool {
	if oldPodSpec == nil {
		return false
	}
	for _, c := range oldPodSpec.Containers {
		for _, rule := range c.RestartPolicyRules {
			if rule.Action == core.ContainerRestartRuleActionRestartAllContainers {
				return true
			}
		}
	}
	for _, c := range oldPodSpec.InitContainers {
		for _, rule := range c.RestartPolicyRules {
			if rule.Action == core.ContainerRestartRuleActionRestartAllContainers {
				return true
			}
		}
		// This feature also allows sidecar containers to have rules.
		if c.RestartPolicy != nil && *c.RestartPolicy == core.ContainerRestartPolicyAlways && len(c.RestartPolicyRules) > 0 {
			return true
		}
	}
	return false
}

// hasRestartContainerForNonSidecarInitContainer returns true if any non-sidecar init container
// has a RestartContainer resize policy.
func hasRestartContainerForNonSidecarInitContainer(spec *core.PodSpec) bool {
	if spec == nil {
		return false
	}
	for _, c := range spec.InitContainers {
		if !podutil.IsRestartableInitContainer(&c) {
			for _, p := range c.ResizePolicy {
				if p.RestartPolicy == core.RestartContainer {
					return true
				}
			}
		}
	}
	return false
}

// seccompProfileOCIInUse returns true if the pod spec or any of its containers
// uses a seccomp profile of type OCI.
func seccompProfileOCIInUse(podSpec *core.PodSpec) bool {
	if podSpec == nil {
		return false
	}
	usesOCI := func(p *core.SeccompProfile) bool {
		return p != nil && p.Type == core.SeccompProfileTypeOCI
	}
	if podSpec.SecurityContext != nil && usesOCI(podSpec.SecurityContext.SeccompProfile) {
		return true
	}
	var inUse bool
	podutil.VisitContainers(podSpec, podutil.AllContainers, func(c *core.Container, containerType podutil.ContainerType) bool {
		if c.SecurityContext != nil && usesOCI(c.SecurityContext.SeccompProfile) {
			inUse = true
			return false
		}
		return true
	})
	return inUse
}
