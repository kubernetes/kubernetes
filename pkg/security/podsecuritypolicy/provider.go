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
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/util/maps"
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
func (s *simpleProvider) CreatePodSecurityContext(pod *api.Pod) (*api.PodSecurityContext, map[string]string, error) {
	sc := securitycontext.NewPodSecurityContextMutator(pod.Spec.SecurityContext)
	annotations := maps.CopySS(pod.Annotations)

	if sc.SupplementalGroups() == nil {
		supGroups, err := s.strategies.SupplementalGroupStrategy.Generate(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SetSupplementalGroups(supGroups)
	}

	if sc.FSGroup() == nil {
		fsGroup, err := s.strategies.FSGroupStrategy.GenerateSingle(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SetFSGroup(fsGroup)
	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, nil)
		if err != nil {
			return nil, nil, err
		}
		sc.SetSELinuxOptions(seLinux)
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
	return sc.PodSecurityContext(), annotations, nil
}

// Create a SecurityContext based on the given constraints.  If a setting is already set on the
// container's security context then it will not be changed.  Validation should be used after
// the context is created to ensure it complies with the required restrictions.
func (s *simpleProvider) CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, map[string]string, error) {
	sc := securitycontext.NewEffectiveContainerSecurityContextMutator(
		securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext),
		securitycontext.NewContainerSecurityContextMutator(container.SecurityContext),
	)

	annotations := maps.CopySS(pod.Annotations)

	if sc.RunAsUser() == nil {
		uid, err := s.strategies.RunAsUserStrategy.Generate(pod, container)
		if err != nil {
			return nil, nil, err
		}
		sc.SetRunAsUser(uid)
	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, container)
		if err != nil {
			return nil, nil, err
		}
		sc.SetSELinuxOptions(seLinux)
	}

	annotations, err := s.strategies.AppArmorStrategy.Generate(annotations, container)
	if err != nil {
		return nil, nil, err
	}

	// if we're using the non-root strategy set the marker that this container should not be
	// run as root which will signal to the kubelet to do a final check either on the runAsUser
	// or, if runAsUser is not set, the image UID will be checked.
	if sc.RunAsNonRoot() == nil && sc.RunAsUser() == nil && s.psp.Spec.RunAsUser.Rule == extensions.RunAsUserStrategyMustRunAsNonRoot {
		nonRoot := true
		sc.SetRunAsNonRoot(&nonRoot)
	}

	caps, err := s.strategies.CapabilitiesStrategy.Generate(pod, container)
	if err != nil {
		return nil, nil, err
	}
	sc.SetCapabilities(caps)

	// if the PSP requires a read only root filesystem and the container has not made a specific
	// request then default ReadOnlyRootFilesystem to true.
	if s.psp.Spec.ReadOnlyRootFilesystem && sc.ReadOnlyRootFilesystem() == nil {
		readOnlyRootFS := true
		sc.SetReadOnlyRootFilesystem(&readOnlyRootFS)
	}

	// if the PSP sets DefaultAllowPrivilegeEscalation and the container security context
	// allowPrivilegeEscalation is not set, then default to that set by the PSP.
	if s.psp.Spec.DefaultAllowPrivilegeEscalation != nil && sc.AllowPrivilegeEscalation() == nil {
		sc.SetAllowPrivilegeEscalation(s.psp.Spec.DefaultAllowPrivilegeEscalation)
	}

	// if the PSP sets psp.AllowPrivilegeEscalation to false set that as the default
	if !s.psp.Spec.AllowPrivilegeEscalation && sc.AllowPrivilegeEscalation() == nil {
		sc.SetAllowPrivilegeEscalation(&s.psp.Spec.AllowPrivilegeEscalation)
	}

	return sc.ContainerSecurityContext(), annotations, nil
}

// Ensure a pod's SecurityContext is in compliance with the given constraints.
func (s *simpleProvider) ValidatePodSecurityContext(pod *api.Pod, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	sc := securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext)

	fsGroups := []int64{}
	if fsGroup := sc.FSGroup(); fsGroup != nil {
		fsGroups = append(fsGroups, *fsGroup)
	}
	allErrs = append(allErrs, s.strategies.FSGroupStrategy.Validate(pod, fsGroups)...)
	allErrs = append(allErrs, s.strategies.SupplementalGroupStrategy.Validate(pod, sc.SupplementalGroups())...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidatePod(pod)...)

	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(fldPath.Child("seLinuxOptions"), pod, nil, sc.SELinuxOptions())...)

	if !s.psp.Spec.HostNetwork && sc.HostNetwork() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), sc.HostNetwork(), "Host network is not allowed to be used"))
	}

	if !s.psp.Spec.HostPID && sc.HostPID() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), sc.HostPID(), "Host PID is not allowed to be used"))
	}

	if !s.psp.Spec.HostIPC && sc.HostIPC() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), sc.HostIPC(), "Host IPC is not allowed to be used"))
	}

	allErrs = append(allErrs, s.strategies.SysctlsStrategy.Validate(pod)...)

	// TODO(tallclair): ValidatePodSecurityContext should be renamed to ValidatePod since its scope
	// is not limited to the PodSecurityContext.
	if len(pod.Spec.Volumes) > 0 {
		allowsAllVolumeTypes := psputil.PSPAllowsAllVolumes(s.psp)
		allowedVolumes := psputil.FSTypeToStringSet(s.psp.Spec.Volumes)
		for i, v := range pod.Spec.Volumes {
			fsType, err := psputil.GetVolumeFSType(v)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "volumes").Index(i), string(fsType), err.Error()))
				continue
			}

			if !allowsAllVolumeTypes && !allowedVolumes.Has(string(fsType)) {
				allErrs = append(allErrs, field.Invalid(
					field.NewPath("spec", "volumes").Index(i), string(fsType),
					fmt.Sprintf("%s volumes are not allowed to be used", string(fsType))))
				continue
			}

			if fsType == extensions.HostPath {
				if !psputil.AllowsHostVolumePath(s.psp, v.HostPath.Path) {
					allErrs = append(allErrs, field.Invalid(
						field.NewPath("spec", "volumes").Index(i).Child("hostPath", "pathPrefix"), v.HostPath.Path,
						fmt.Sprintf("is not allowed to be used")))
				}
			}
		}
	}

	return allErrs
}

// Ensure a container's SecurityContext is in compliance with the given constraints
func (s *simpleProvider) ValidateContainerSecurityContext(pod *api.Pod, container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	podSC := securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext)
	sc := securitycontext.NewEffectiveContainerSecurityContextAccessor(podSC, securitycontext.NewContainerSecurityContextMutator(container.SecurityContext))

	allErrs = append(allErrs, s.strategies.RunAsUserStrategy.Validate(fldPath.Child("securityContext"), pod, container, sc.RunAsNonRoot(), sc.RunAsUser())...)
	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(fldPath.Child("seLinuxOptions"), pod, container, sc.SELinuxOptions())...)
	allErrs = append(allErrs, s.strategies.AppArmorStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidateContainer(pod, container)...)

	privileged := sc.Privileged()
	if !s.psp.Spec.Privileged && privileged != nil && *privileged {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("privileged"), *privileged, "Privileged containers are not allowed"))
	}

	allErrs = append(allErrs, s.strategies.CapabilitiesStrategy.Validate(pod, container, sc.Capabilities())...)

	if !s.psp.Spec.HostNetwork && podSC.HostNetwork() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), podSC.HostNetwork(), "Host network is not allowed to be used"))
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

	if !s.psp.Spec.HostPID && podSC.HostPID() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), podSC.HostPID(), "Host PID is not allowed to be used"))
	}

	if !s.psp.Spec.HostIPC && podSC.HostIPC() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), podSC.HostIPC(), "Host IPC is not allowed to be used"))
	}

	if s.psp.Spec.ReadOnlyRootFilesystem {
		readOnly := sc.ReadOnlyRootFilesystem()
		if readOnly == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), readOnly, "ReadOnlyRootFilesystem may not be nil and must be set to true"))
		} else if !*readOnly {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), *readOnly, "ReadOnlyRootFilesystem must be set to true"))
		}
	}

	allowEscalation := sc.AllowPrivilegeEscalation()
	if !s.psp.Spec.AllowPrivilegeEscalation && allowEscalation == nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("allowPrivilegeEscalation"), allowEscalation, "Allowing privilege escalation for containers is not allowed"))
	}

	if !s.psp.Spec.AllowPrivilegeEscalation && allowEscalation != nil && *allowEscalation {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("allowPrivilegeEscalation"), *allowEscalation, "Allowing privilege escalation for containers is not allowed"))
	}

	return allErrs
}

// hasHostPort checks the port definitions on the container for HostPort > 0.
func (s *simpleProvider) hasInvalidHostPort(container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, cp := range container.Ports {
		if cp.HostPort > 0 && !s.isValidHostPort(int(cp.HostPort)) {
			detail := fmt.Sprintf("Host port %d is not allowed to be used. Allowed ports: [%s]", cp.HostPort, hostPortRangesToString(s.psp.Spec.HostPorts))
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

func hostPortRangesToString(ranges []extensions.HostPortRange) string {
	formattedString := ""
	if ranges != nil {
		strRanges := []string{}
		for _, r := range ranges {
			if r.Min == r.Max {
				strRanges = append(strRanges, fmt.Sprintf("%d", r.Min))
			} else {
				strRanges = append(strRanges, fmt.Sprintf("%d-%d", r.Min, r.Max))
			}
		}
		formattedString = strings.Join(strRanges, ",")
	}
	return formattedString
}
