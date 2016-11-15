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

package podsecuritypolicy

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/pkg/util/maps"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// used to pass in the field being validated for reusable group strategies so they
// can create informative error messages.
const (
	fsGroupField            = "fsGroup"
	supplementalGroupsField = "supplementalGroups"
)

// simpleProvider is the default implementation of Provider.
type simpleProvider struct {
	psp        *extensions.PodSecurityPolicy
	strategies *ProviderStrategies
}

// ensure we implement the interface correctly.
var _ Provider = &simpleProvider{}

// NewSimpleProvider creates a new Provider instance.
func NewSimpleProvider(psp *extensions.PodSecurityPolicy, namespace string, strategyFactory StrategyFactory) (Provider, error) {
	if psp == nil {
		return nil, fmt.Errorf("NewSimpleProvider requires a PodSecurityPolicy")
	}
	if strategyFactory == nil {
		return nil, fmt.Errorf("NewSimpleProvider requires a StrategyFactory")
	}

	strategies, err := strategyFactory.CreateStrategies(psp, namespace)
	if err != nil {
		return nil, err
	}

	return &simpleProvider{
		psp:        psp,
		strategies: strategies,
	}, nil
}

// Create a PodSecurityContext based on the given constraints.  If a setting is already set
// on the PodSecurityContext it will not be changed.  Validate should be used after the context
// is created to ensure it complies with the required restrictions.
//
// NOTE: this method works on a copy of the PodSecurityContext.  It is up to the caller to
// apply the PSC if validation passes.
func (s *simpleProvider) CreatePodSecurityContext(pod *api.Pod) (*api.PodSecurityContext, map[string]string, error) {
	var sc *api.PodSecurityContext = nil
	if pod.Spec.SecurityContext != nil {
		// work with a copy
		copy := *pod.Spec.SecurityContext
		sc = &copy
	} else {
		sc = &api.PodSecurityContext{}
	}
	annotations := maps.CopySS(pod.Annotations)

	if len(sc.SupplementalGroups) == 0 {
		supGroups, err := s.strategies.SupplementalGroupStrategy.Generate(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SupplementalGroups = supGroups
	}

	if sc.FSGroup == nil {
		fsGroup, err := s.strategies.FSGroupStrategy.GenerateSingle(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.FSGroup = fsGroup
	}

	if sc.SELinuxOptions == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, nil)
		if err != nil {
			return nil, nil, err
		}
		sc.SELinuxOptions = seLinux
	}

	// This is only generated on the pod level.  Containers inherit the pod's profile.  If the
	// container has a specific profile set then it will be caught in the validation step.
	seccompProfile, err := s.strategies.SeccompStrategy.Generate(annotations, pod)
	if err != nil {
		return nil, nil, err
	}
	if seccompProfile != "" {
		if annotations == nil {
			annotations = map[string]string{}
		}
		annotations[api.SeccompPodAnnotationKey] = seccompProfile
	}
	return sc, annotations, nil
}

// Create a SecurityContext based on the given constraints.  If a setting is already set on the
// container's security context then it will not be changed.  Validation should be used after
// the context is created to ensure it complies with the required restrictions.
//
// NOTE: this method works on a copy of the SC of the container.  It is up to the caller to apply
// the SC if validation passes.
func (s *simpleProvider) CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, map[string]string, error) {
	var sc *api.SecurityContext = nil
	if container.SecurityContext != nil {
		// work with a copy of the original
		copy := *container.SecurityContext
		sc = &copy
	} else {
		sc = &api.SecurityContext{}
	}
	annotations := maps.CopySS(pod.Annotations)

	if sc.RunAsUser == nil {
		uid, err := s.strategies.RunAsUserStrategy.Generate(pod, container)
		if err != nil {
			return nil, nil, err
		}
		sc.RunAsUser = uid
	}

	if sc.SELinuxOptions == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, container)
		if err != nil {
			return nil, nil, err
		}
		sc.SELinuxOptions = seLinux
	}

	annotations, err := s.strategies.AppArmorStrategy.Generate(annotations, container)
	if err != nil {
		return nil, nil, err
	}

	if sc.Privileged == nil {
		priv := false
		sc.Privileged = &priv
	}

	// if we're using the non-root strategy set the marker that this container should not be
	// run as root which will signal to the kubelet to do a final check either on the runAsUser
	// or, if runAsUser is not set, the image UID will be checked.
	if s.psp.Spec.RunAsUser.Rule == extensions.RunAsUserStrategyMustRunAsNonRoot {
		nonRoot := true
		sc.RunAsNonRoot = &nonRoot
	}

	caps, err := s.strategies.CapabilitiesStrategy.Generate(pod, container)
	if err != nil {
		return nil, nil, err
	}
	sc.Capabilities = caps

	// if the PSP requires a read only root filesystem and the container has not made a specific
	// request then default ReadOnlyRootFilesystem to true.
	if s.psp.Spec.ReadOnlyRootFilesystem && sc.ReadOnlyRootFilesystem == nil {
		readOnlyRootFS := true
		sc.ReadOnlyRootFilesystem = &readOnlyRootFS
	}

	return sc, annotations, nil
}

// Ensure a pod's SecurityContext is in compliance with the given constraints.
func (s *simpleProvider) ValidatePodSecurityContext(pod *api.Pod, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if pod.Spec.SecurityContext == nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("securityContext"), pod.Spec.SecurityContext, "No security context is set"))
		return allErrs
	}

	fsGroups := []int64{}
	if pod.Spec.SecurityContext.FSGroup != nil {
		fsGroups = append(fsGroups, *pod.Spec.SecurityContext.FSGroup)
	}
	allErrs = append(allErrs, s.strategies.FSGroupStrategy.Validate(pod, fsGroups)...)
	allErrs = append(allErrs, s.strategies.SupplementalGroupStrategy.Validate(pod, pod.Spec.SecurityContext.SupplementalGroups)...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidatePod(pod)...)

	// make a dummy container context to reuse the selinux strategies
	container := &api.Container{
		Name: pod.Name,
		SecurityContext: &api.SecurityContext{
			SELinuxOptions: pod.Spec.SecurityContext.SELinuxOptions,
		},
	}
	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(pod, container)...)

	if !s.psp.Spec.HostNetwork && pod.Spec.SecurityContext.HostNetwork {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), pod.Spec.SecurityContext.HostNetwork, "Host network is not allowed to be used"))
	}

	if !s.psp.Spec.HostPID && pod.Spec.SecurityContext.HostPID {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), pod.Spec.SecurityContext.HostPID, "Host PID is not allowed to be used"))
	}

	if !s.psp.Spec.HostIPC && pod.Spec.SecurityContext.HostIPC {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), pod.Spec.SecurityContext.HostIPC, "Host IPC is not allowed to be used"))
	}

	allErrs = append(allErrs, s.strategies.SysctlsStrategy.Validate(pod)...)

	// TODO(timstclair): ValidatePodSecurityContext should be renamed to ValidatePod since its scope
	// is not limited to the PodSecurityContext.
	if len(pod.Spec.Volumes) > 0 && !psputil.PSPAllowsAllVolumes(s.psp) {
		allowedVolumes := psputil.FSTypeToStringSet(s.psp.Spec.Volumes)
		for i, v := range pod.Spec.Volumes {
			fsType, err := psputil.GetVolumeFSType(v)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "volumes").Index(i), string(fsType), err.Error()))
				continue
			}

			if !allowedVolumes.Has(string(fsType)) {
				allErrs = append(allErrs, field.Invalid(
					field.NewPath("spec", "volumes").Index(i), string(fsType),
					fmt.Sprintf("%s volumes are not allowed to be used", string(fsType))))
			}
		}
	}

	return allErrs
}

// Ensure a container's SecurityContext is in compliance with the given constraints
func (s *simpleProvider) ValidateContainerSecurityContext(pod *api.Pod, container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if container.SecurityContext == nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("securityContext"), container.SecurityContext, "No security context is set"))
		return allErrs
	}

	sc := container.SecurityContext
	allErrs = append(allErrs, s.strategies.RunAsUserStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.strategies.AppArmorStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidateContainer(pod, container)...)

	if !s.psp.Spec.Privileged && *sc.Privileged {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("privileged"), *sc.Privileged, "Privileged containers are not allowed"))
	}

	allErrs = append(allErrs, s.strategies.CapabilitiesStrategy.Validate(pod, container)...)

	if !s.psp.Spec.HostNetwork && pod.Spec.SecurityContext.HostNetwork {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), pod.Spec.SecurityContext.HostNetwork, "Host network is not allowed to be used"))
	}

	containersPath := fldPath.Child("containers")
	for idx, c := range pod.Spec.Containers {
		idxPath := containersPath.Index(idx)
		allErrs = append(allErrs, s.hasInvalidHostPort(&c, idxPath)...)
	}

	containersPath = fldPath.Child("initContainers")
	for idx, c := range pod.Spec.InitContainers {
		idxPath := containersPath.Index(idx)
		allErrs = append(allErrs, s.hasInvalidHostPort(&c, idxPath)...)
	}

	if !s.psp.Spec.HostPID && pod.Spec.SecurityContext.HostPID {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), pod.Spec.SecurityContext.HostPID, "Host PID is not allowed to be used"))
	}

	if !s.psp.Spec.HostIPC && pod.Spec.SecurityContext.HostIPC {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), pod.Spec.SecurityContext.HostIPC, "Host IPC is not allowed to be used"))
	}

	if s.psp.Spec.ReadOnlyRootFilesystem {
		if sc.ReadOnlyRootFilesystem == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), sc.ReadOnlyRootFilesystem, "ReadOnlyRootFilesystem may not be nil and must be set to true"))
		} else if !*sc.ReadOnlyRootFilesystem {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), *sc.ReadOnlyRootFilesystem, "ReadOnlyRootFilesystem must be set to true"))
		}
	}

	return allErrs
}

// hasHostPort checks the port definitions on the container for HostPort > 0.
func (s *simpleProvider) hasInvalidHostPort(container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, cp := range container.Ports {
		if cp.HostPort > 0 && !s.isValidHostPort(int(cp.HostPort)) {
			detail := fmt.Sprintf("Host port %d is not allowed to be used.  Allowed ports: %v", cp.HostPort, s.psp.Spec.HostPorts)
			allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPort"), cp.HostPort, detail))
		}
	}
	return allErrs
}

// isValidHostPort returns true if the port falls in any range allowed by the PSP.
func (s *simpleProvider) isValidHostPort(port int) bool {
	for _, hostPortRange := range s.psp.Spec.HostPorts {
		if port >= hostPortRange.Min && port <= hostPortRange.Max {
			return true
		}
	}
	return false
}

// Get the name of the PSP that this provider was initialized with.
func (s *simpleProvider) GetPSPName() string {
	return s.psp.Name
}
