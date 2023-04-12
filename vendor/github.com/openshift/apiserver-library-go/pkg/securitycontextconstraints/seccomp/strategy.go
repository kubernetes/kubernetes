package seccomp

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	allowAnyProfile = "*"
)

// Strategy defines the interface for all seccomp constraint strategies.
type SeccompStrategy interface {
	// Generate returns a profile based on constraint rules.
	Generate(annotations map[string]string, pod *api.Pod) (string, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidatePod(pod *api.Pod) field.ErrorList
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList
}

type strategy struct {
	allowedProfiles []string
	// does the strategy allow any profile (wildcard)
	allowAnyProfile       bool
	runtimeDefaultAllowed bool
}

var _ SeccompStrategy = &strategy{}

// NewStrategy creates a new strategy that enforces seccomp profile constraints.
func NewSeccompStrategy(allowedProfiles []string) SeccompStrategy {
	allowAny := false
	allowed := make([]string, 0, len(allowedProfiles))
	runtimeDefaultAllowed := false

	for _, p := range allowedProfiles {
		if p == allowAnyProfile {
			allowAny = true
			continue
		}
		// With the graduation of seccomp to GA we automatically convert
		// the deprecated seccomp profile `docker/default` to `runtime/default`.
		// This means that we now have to automatically allow `runtime/default`
		// if a user specifies `docker/default` and vice versa in an SCC.
		if p == v1.DeprecatedSeccompProfileDockerDefault || p == v1.SeccompProfileRuntimeDefault {
			runtimeDefaultAllowed = true
		}
		allowed = append(allowed, p)
	}

	return &strategy{
		allowedProfiles:       allowed,
		allowAnyProfile:       allowAny,
		runtimeDefaultAllowed: runtimeDefaultAllowed,
	}
}

// Generate returns a profile based on constraint rules.
func (s *strategy) Generate(podAnnotations map[string]string, pod *api.Pod) (string, error) {
	if podAnnotations[api.SeccompPodAnnotationKey] != "" {
		// Profile already set, nothing to do.
		return podAnnotations[api.SeccompPodAnnotationKey], nil
	}
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SeccompProfile != nil {
		// Profile field already set, translate to annotation.
		return seccompAnnotationForField(pod.Spec.SecurityContext.SeccompProfile), nil
	}

	// return the first non-wildcard profile
	if len(s.allowedProfiles) > 0 {
		return s.allowedProfiles[0], nil
	}

	return "", nil
}

// ValidatePod ensures that the specified values on the pod fall within the range
// of the strategy.
func (s *strategy) ValidatePod(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}
	podSpecFieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.SeccompPodAnnotationKey)
	podProfile := pod.Annotations[api.SeccompPodAnnotationKey]
	// if the annotation is not set, see if the field is set and derive the corresponding annotation value
	// We are keeping annotations for backward compatibility - in case the pod is
	// running on an older node.
	if len(podProfile) == 0 && pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SeccompProfile != nil {
		podProfile = seccompAnnotationForField(pod.Spec.SecurityContext.SeccompProfile)
	}

	if err := s.validateProfile(podSpecFieldPath, podProfile); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}

// ValidateContainer ensures that the specified values on the container fall within
// the range of the strategy.
func (s *strategy) ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}
	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.SeccompContainerAnnotationKeyPrefix + container.Name)
	containerProfile := profileForContainer(pod, container)

	if err := s.validateProfile(fieldPath, containerProfile); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}

// validateProfile checks if profile is in allowedProfiles or if allowedProfiles
// contains the wildcard.
func (s *strategy) validateProfile(fldPath *field.Path, profile string) *field.Error {
	if !s.allowAnyProfile && len(s.allowedProfiles) == 0 && profile != "" {
		return field.Forbidden(fldPath, "seccomp may not be set")
	}

	// for backwards compatibility and SCCs without a defined list of allowed profiles.
	// If a SCC does not have allowedProfiles set then we should allow an empty profile.
	// This will mean that the runtime default is used.
	if len(s.allowedProfiles) == 0 && profile == "" {
		return nil
	}

	if s.allowAnyProfile {
		return nil
	}

	for _, p := range s.allowedProfiles {
		if profile == p {
			return nil
		}

		// With the graduation of seccomp to GA we automatically convert
		// the deprecated seccomp profile `docker/default` to `runtime/default`.
		// This means that we now have to automatically allow `runtime/default`
		// if a user specifies `docker/default` and vice versa in an SCC.
		if s.runtimeDefaultAllowed &&
			(profile == v1.DeprecatedSeccompProfileDockerDefault ||
				profile == v1.SeccompProfileRuntimeDefault) {
			return nil
		}
	}

	return field.Forbidden(fldPath, fmt.Sprintf("%s is not an allowed seccomp profile. Valid values are %v", profile, s.allowedProfiles))
}

// profileForContainer returns the container profile if set, otherwise the pod profile.
func profileForContainer(pod *api.Pod, container *api.Container) string {
	if container.SecurityContext != nil && container.SecurityContext.SeccompProfile != nil {
		// derive the annotation value from the container field
		return seccompAnnotationForField(container.SecurityContext.SeccompProfile)
	}
	containerProfile, ok := pod.Annotations[api.SeccompContainerAnnotationKeyPrefix+container.Name]
	if ok {
		// return the existing container annotation
		return containerProfile
	}
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SeccompProfile != nil {
		// derive the annotation value from the pod field
		return seccompAnnotationForField(pod.Spec.SecurityContext.SeccompProfile)
	}
	// return the existing pod annotation
	return pod.Annotations[api.SeccompPodAnnotationKey]
}

// seccompAnnotationForField takes a pod seccomp profile field and returns the
// converted annotation value.
// DEPRECATED: this is originally from k8s.io/kubernetes/pkg/api pod module which has
// been removed in upstream: https://github.com/kubernetes/kubernetes/pull/114947/files.
// TODO(auth team): remove once we stop handling the annotation.
func seccompAnnotationForField(field *api.SeccompProfile) string {
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
