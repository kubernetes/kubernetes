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

	corev1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/pods"
	"k8s.io/kubernetes/pkg/features"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// simpleProvider is the default implementation of Provider.
type simpleProvider struct {
	psp        *policy.PodSecurityPolicy
	strategies *ProviderStrategies
}

// ensure we implement the interface correctly.
var _ Provider = &simpleProvider{}

// NewSimpleProvider creates a new Provider instance.
func NewSimpleProvider(psp *policy.PodSecurityPolicy, namespace string, strategyFactory StrategyFactory) (Provider, error) {
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

// MutatePod sets the default values of the required but not filled fields.
// Validation should be used after the context is defaulted to ensure it
// complies with the required restrictions.
func (s *simpleProvider) MutatePod(pod *api.Pod) error {
	sc := securitycontext.NewPodSecurityContextMutator(pod.Spec.SecurityContext)

	if sc.SupplementalGroups() == nil {
		supGroups, err := s.strategies.SupplementalGroupStrategy.Generate(pod)
		if err != nil {
			return err
		}
		sc.SetSupplementalGroups(supGroups)
	}

	if sc.FSGroup() == nil {
		fsGroup, err := s.strategies.FSGroupStrategy.GenerateSingle(pod)
		if err != nil {
			return err
		}
		sc.SetFSGroup(fsGroup)
	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, nil)
		if err != nil {
			return err
		}
		sc.SetSELinuxOptions(seLinux)
	}

	// This is only generated on the pod level.  Containers inherit the pod's profile.  If the
	// container has a specific profile set then it will be caught in the validation step.
	seccompProfile, err := s.strategies.SeccompStrategy.Generate(pod.Annotations, pod)
	if err != nil {
		return err
	}
	if seccompProfile != "" {
		if pod.Annotations == nil {
			pod.Annotations = map[string]string{}
		}
		pod.Annotations[api.SeccompPodAnnotationKey] = seccompProfile
	}

	pod.Spec.SecurityContext = sc.PodSecurityContext()

	if s.psp.Spec.RuntimeClass != nil && pod.Spec.RuntimeClassName == nil {
		pod.Spec.RuntimeClassName = s.psp.Spec.RuntimeClass.DefaultRuntimeClassName
	}

	var retErr error
	podutil.VisitContainers(&pod.Spec, func(c *api.Container) bool {
		retErr = s.mutateContainer(pod, c)
		if retErr != nil {
			return false
		}
		return true
	})

	return retErr
}

// mutateContainer sets the default values of the required but not filled fields.
// It modifies the SecurityContext of the container and annotations of the pod. Validation should
// be used after the context is defaulted to ensure it complies with the required restrictions.
func (s *simpleProvider) mutateContainer(pod *api.Pod, container *api.Container) error {
	sc := securitycontext.NewEffectiveContainerSecurityContextMutator(
		securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext),
		securitycontext.NewContainerSecurityContextMutator(container.SecurityContext),
	)

	if sc.RunAsUser() == nil {
		uid, err := s.strategies.RunAsUserStrategy.Generate(pod, container)
		if err != nil {
			return err
		}
		sc.SetRunAsUser(uid)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) {
		if sc.RunAsGroup() == nil {
			gid, err := s.strategies.RunAsGroupStrategy.GenerateSingle(pod)
			if err != nil {
				return err
			}
			sc.SetRunAsGroup(gid)
		}

	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.strategies.SELinuxStrategy.Generate(pod, container)
		if err != nil {
			return err
		}
		sc.SetSELinuxOptions(seLinux)
	}

	annotations, err := s.strategies.AppArmorStrategy.Generate(pod.Annotations, container)
	if err != nil {
		return err
	}

	// if we're using the non-root strategy set the marker that this container should not be
	// run as root which will signal to the kubelet to do a final check either on the runAsUser
	// or, if runAsUser is not set, the image UID will be checked.
	if sc.RunAsNonRoot() == nil && sc.RunAsUser() == nil && s.psp.Spec.RunAsUser.Rule == policy.RunAsUserStrategyMustRunAsNonRoot {
		nonRoot := true
		sc.SetRunAsNonRoot(&nonRoot)
	}

	caps, err := s.strategies.CapabilitiesStrategy.Generate(pod, container)
	if err != nil {
		return err
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

	// if the PSP sets psp.AllowPrivilegeEscalation to false, set that as the default
	if !*s.psp.Spec.AllowPrivilegeEscalation && sc.AllowPrivilegeEscalation() == nil {
		sc.SetAllowPrivilegeEscalation(s.psp.Spec.AllowPrivilegeEscalation)
	}

	pod.Annotations = annotations
	container.SecurityContext = sc.ContainerSecurityContext()

	return nil
}

// ValidatePod ensure a pod is in compliance with the given constraints.
func (s *simpleProvider) ValidatePod(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}

	sc := securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext)
	scPath := field.NewPath("spec", "securityContext")

	var fsGroups []int64
	if fsGroup := sc.FSGroup(); fsGroup != nil {
		fsGroups = []int64{*fsGroup}
	}
	allErrs = append(allErrs, s.strategies.FSGroupStrategy.Validate(scPath.Child("fsGroup"), pod, fsGroups)...)
	allErrs = append(allErrs, s.strategies.SupplementalGroupStrategy.Validate(scPath.Child("supplementalGroups"), pod, sc.SupplementalGroups())...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidatePod(pod)...)

	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(scPath.Child("seLinuxOptions"), pod, nil, sc.SELinuxOptions())...)

	if !s.psp.Spec.HostNetwork && sc.HostNetwork() {
		allErrs = append(allErrs, field.Invalid(scPath.Child("hostNetwork"), sc.HostNetwork(), "Host network is not allowed to be used"))
	}

	if !s.psp.Spec.HostPID && sc.HostPID() {
		allErrs = append(allErrs, field.Invalid(scPath.Child("hostPID"), sc.HostPID(), "Host PID is not allowed to be used"))
	}

	if !s.psp.Spec.HostIPC && sc.HostIPC() {
		allErrs = append(allErrs, field.Invalid(scPath.Child("hostIPC"), sc.HostIPC(), "Host IPC is not allowed to be used"))
	}

	allErrs = append(allErrs, s.strategies.SysctlsStrategy.Validate(pod)...)

	allErrs = append(allErrs, s.validatePodVolumes(pod)...)

	if s.psp.Spec.RuntimeClass != nil {
		allErrs = append(allErrs, validateRuntimeClassName(pod.Spec.RuntimeClassName, s.psp.Spec.RuntimeClass.AllowedRuntimeClassNames)...)
	}

	pods.VisitContainersWithPath(&pod.Spec, func(c *api.Container, p *field.Path) bool {
		allErrs = append(allErrs, s.validateContainer(pod, c, p)...)
		return true
	})

	return allErrs
}

func (s *simpleProvider) validatePodVolumes(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}

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

			switch fsType {
			case policy.HostPath:
				allows, mustBeReadOnly := psputil.AllowsHostVolumePath(s.psp, v.HostPath.Path)
				if !allows {
					allErrs = append(allErrs, field.Invalid(
						field.NewPath("spec", "volumes").Index(i).Child("hostPath", "pathPrefix"), v.HostPath.Path,
						fmt.Sprintf("is not allowed to be used")))
				} else if mustBeReadOnly {
					// Ensure all the VolumeMounts that use this volume are read-only
					pods.VisitContainersWithPath(&pod.Spec, func(c *api.Container, p *field.Path) bool {
						for i, cv := range c.VolumeMounts {
							if cv.Name == v.Name && !cv.ReadOnly {
								allErrs = append(allErrs, field.Invalid(p.Child("volumeMounts").Index(i).Child("readOnly"), cv.ReadOnly, "must be read-only"))
							}
						}
						return true
					})
				}

			case policy.FlexVolume:
				if len(s.psp.Spec.AllowedFlexVolumes) > 0 {
					found := false
					driver := v.FlexVolume.Driver
					for _, allowedFlexVolume := range s.psp.Spec.AllowedFlexVolumes {
						if driver == allowedFlexVolume.Driver {
							found = true
							break
						}
					}
					if !found {
						allErrs = append(allErrs,
							field.Invalid(field.NewPath("spec", "volumes").Index(i).Child("driver"), driver,
								"Flexvolume driver is not allowed to be used"))
					}
				}

			case policy.CSI:
				if utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
					if len(s.psp.Spec.AllowedCSIDrivers) > 0 {
						found := false
						driver := v.CSI.Driver
						for _, allowedCSIDriver := range s.psp.Spec.AllowedCSIDrivers {
							if driver == allowedCSIDriver.Name {
								found = true
								break
							}
						}
						if !found {
							allErrs = append(allErrs,
								field.Invalid(field.NewPath("spec", "volumes").Index(i).Child("csi", "driver"), driver,
									"Inline CSI driver is not allowed to be used"))
						}
					}
				}
			}
		}
	}

	return allErrs
}

// Ensure a container's SecurityContext is in compliance with the given constraints
func (s *simpleProvider) validateContainer(pod *api.Pod, container *api.Container, containerPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	podSC := securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext)
	sc := securitycontext.NewEffectiveContainerSecurityContextAccessor(podSC, securitycontext.NewContainerSecurityContextMutator(container.SecurityContext))

	scPath := containerPath.Child("securityContext")
	allErrs = append(allErrs, s.strategies.RunAsUserStrategy.Validate(scPath, pod, container, sc.RunAsNonRoot(), sc.RunAsUser())...)

	if utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) {
		var runAsGroups []int64
		if sc.RunAsGroup() != nil {
			runAsGroups = []int64{*sc.RunAsGroup()}
		}
		allErrs = append(allErrs, s.strategies.RunAsGroupStrategy.Validate(scPath, pod, runAsGroups)...)
	}
	allErrs = append(allErrs, s.strategies.SELinuxStrategy.Validate(scPath.Child("seLinuxOptions"), pod, container, sc.SELinuxOptions())...)
	allErrs = append(allErrs, s.strategies.AppArmorStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.strategies.SeccompStrategy.ValidateContainer(pod, container)...)

	privileged := sc.Privileged()
	if !s.psp.Spec.Privileged && privileged != nil && *privileged {
		allErrs = append(allErrs, field.Invalid(scPath.Child("privileged"), *privileged, "Privileged containers are not allowed"))
	}

	procMount := sc.ProcMount()
	allowedProcMounts := s.psp.Spec.AllowedProcMountTypes
	if len(allowedProcMounts) == 0 {
		allowedProcMounts = []corev1.ProcMountType{corev1.DefaultProcMount}
	}
	foundProcMountType := false
	for _, pm := range allowedProcMounts {
		if string(pm) == string(procMount) {
			foundProcMountType = true
		}
	}

	if !foundProcMountType {
		allErrs = append(allErrs, field.Invalid(scPath.Child("procMount"), procMount, "ProcMountType is not allowed"))
	}

	allErrs = append(allErrs, s.strategies.CapabilitiesStrategy.Validate(scPath.Child("capabilities"), pod, container, sc.Capabilities())...)

	allErrs = append(allErrs, s.hasInvalidHostPort(container, containerPath)...)

	if s.psp.Spec.ReadOnlyRootFilesystem {
		readOnly := sc.ReadOnlyRootFilesystem()
		if readOnly == nil {
			allErrs = append(allErrs, field.Invalid(scPath.Child("readOnlyRootFilesystem"), readOnly, "ReadOnlyRootFilesystem may not be nil and must be set to true"))
		} else if !*readOnly {
			allErrs = append(allErrs, field.Invalid(scPath.Child("readOnlyRootFilesystem"), *readOnly, "ReadOnlyRootFilesystem must be set to true"))
		}
	}

	allowEscalation := sc.AllowPrivilegeEscalation()
	if !*s.psp.Spec.AllowPrivilegeEscalation && (allowEscalation == nil || *allowEscalation) {
		allErrs = append(allErrs, field.Invalid(scPath.Child("allowPrivilegeEscalation"), allowEscalation, "Allowing privilege escalation for containers is not allowed"))
	}

	return allErrs
}

// hasInvalidHostPort checks whether the port definitions on the container fall outside of the ranges allowed by the PSP.
func (s *simpleProvider) hasInvalidHostPort(container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, cp := range container.Ports {
		if cp.HostPort > 0 && !s.isValidHostPort(cp.HostPort) {
			detail := fmt.Sprintf("Host port %d is not allowed to be used. Allowed ports: [%s]", cp.HostPort, hostPortRangesToString(s.psp.Spec.HostPorts))
			allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPort"), cp.HostPort, detail))
		}
	}
	return allErrs
}

// isValidHostPort returns true if the port falls in any range allowed by the PSP.
func (s *simpleProvider) isValidHostPort(port int32) bool {
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

func hostPortRangesToString(ranges []policy.HostPortRange) string {
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

// validates that the actual RuntimeClassName is contained in the list of valid names.
func validateRuntimeClassName(actual *string, validNames []string) field.ErrorList {
	if actual == nil {
		return nil // An unset RuntimeClassName is always allowed.
	}

	for _, valid := range validNames {
		if valid == policy.AllowAllRuntimeClassNames {
			return nil
		}
		if *actual == valid {
			return nil
		}
	}
	return field.ErrorList{field.Invalid(field.NewPath("spec", "runtimeClassName"), *actual, "")}
}
