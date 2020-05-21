package sccmatching

import (
	"fmt"

	securityv1 "github.com/openshift/api/security/v1"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/capabilities"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/group"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/seccomp"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/selinux"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/user"
	sccutil "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/sysctl"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/util/maps"
)

// used to pass in the field being validated for reusable group strategies so they
// can create informative error messages.
const (
	fsGroupField            = "fsGroup"
	supplementalGroupsField = "supplementalGroups"
)

// simpleProvider is the default implementation of SecurityContextConstraintsProvider
type simpleProvider struct {
	scc                       *securityv1.SecurityContextConstraints
	runAsUserStrategy         user.RunAsUserSecurityContextConstraintsStrategy
	seLinuxStrategy           selinux.SELinuxSecurityContextConstraintsStrategy
	fsGroupStrategy           group.GroupSecurityContextConstraintsStrategy
	supplementalGroupStrategy group.GroupSecurityContextConstraintsStrategy
	capabilitiesStrategy      capabilities.CapabilitiesSecurityContextConstraintsStrategy
	seccompStrategy           seccomp.SeccompStrategy
	sysctlsStrategy           sysctl.SysctlsStrategy
}

// ensure we implement the interface correctly.
var _ SecurityContextConstraintsProvider = &simpleProvider{}

// NewSimpleProvider creates a new SecurityContextConstraintsProvider instance.
func NewSimpleProvider(scc *securityv1.SecurityContextConstraints) (SecurityContextConstraintsProvider, error) {
	if scc == nil {
		return nil, fmt.Errorf("NewSimpleProvider requires a SecurityContextConstraints")
	}

	userStrat, err := createUserStrategy(&scc.RunAsUser)
	if err != nil {
		return nil, err
	}

	seLinuxStrat, err := createSELinuxStrategy(&scc.SELinuxContext)
	if err != nil {
		return nil, err
	}

	fsGroupStrat, err := createFSGroupStrategy(&scc.FSGroup)
	if err != nil {
		return nil, err
	}

	supGroupStrat, err := createSupplementalGroupStrategy(&scc.SupplementalGroups)
	if err != nil {
		return nil, err
	}

	capStrat, err := createCapabilitiesStrategy(scc.DefaultAddCapabilities, scc.RequiredDropCapabilities, scc.AllowedCapabilities)
	if err != nil {
		return nil, err
	}

	seccompStrat, err := createSeccompStrategy(scc.SeccompProfiles)
	if err != nil {
		return nil, err
	}

	sysctlsStrat, err := createSysctlsStrategy(sysctl.SafeSysctlWhitelist(), scc.AllowedUnsafeSysctls, scc.ForbiddenSysctls)
	if err != nil {
		return nil, err
	}

	return &simpleProvider{
		scc:                       scc,
		runAsUserStrategy:         userStrat,
		seLinuxStrategy:           seLinuxStrat,
		fsGroupStrategy:           fsGroupStrat,
		supplementalGroupStrategy: supGroupStrat,
		capabilitiesStrategy:      capStrat,
		seccompStrategy:           seccompStrat,
		sysctlsStrategy:           sysctlsStrat,
	}, nil
}

// Create a PodSecurityContext based on the given constraints.  If a setting is already set
// on the PodSecurityContext it will not be changed.  Validate should be used after the context
// is created to ensure it complies with the required restrictions.
func (s *simpleProvider) CreatePodSecurityContext(pod *api.Pod) (*api.PodSecurityContext, map[string]string, error) {
	sc := securitycontext.NewPodSecurityContextMutator(pod.Spec.SecurityContext)

	annotationsCopy := maps.CopySS(pod.Annotations)

	if sc.SupplementalGroups() == nil {
		supGroups, err := s.supplementalGroupStrategy.Generate(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SetSupplementalGroups(supGroups)
	}

	if sc.FSGroup() == nil {
		fsGroup, err := s.fsGroupStrategy.GenerateSingle(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SetFSGroup(fsGroup)
	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.seLinuxStrategy.Generate(pod, nil)
		if err != nil {
			return nil, nil, err
		}
		sc.SetSELinuxOptions(seLinux)
	}

	// we only generate a seccomp annotation for the entire pod.  Validation
	// will catch any container annotations that are invalid and containers
	// will inherit the pod annotation.
	_, hasPodProfile := pod.Annotations[api.SeccompPodAnnotationKey]
	if !hasPodProfile {
		profile, err := s.seccompStrategy.Generate(pod)
		if err != nil {
			return nil, nil, err
		}

		if profile != "" {
			if annotationsCopy == nil {
				annotationsCopy = map[string]string{}
			}
			annotationsCopy[api.SeccompPodAnnotationKey] = profile
		}
	}

	return sc.PodSecurityContext(), annotationsCopy, nil
}

// Create a SecurityContext based on the given constraints.  If a setting is already set on the
// container's security context then it will not be changed.  Validation should be used after
// the context is created to ensure it complies with the required restrictions.
func (s *simpleProvider) CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, error) {
	sc := securitycontext.NewEffectiveContainerSecurityContextMutator(
		securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext),
		securitycontext.NewContainerSecurityContextMutator(container.SecurityContext),
	)
	if sc.RunAsUser() == nil {
		uid, err := s.runAsUserStrategy.Generate(pod, container)
		if err != nil {
			return nil, err
		}
		sc.SetRunAsUser(uid)
	}

	if sc.SELinuxOptions() == nil {
		seLinux, err := s.seLinuxStrategy.Generate(pod, container)
		if err != nil {
			return nil, err
		}
		sc.SetSELinuxOptions(seLinux)
	}

	// if we're using the non-root strategy set the marker that this container should not be
	// run as root which will signal to the kubelet to do a final check either on the runAsUser
	// or, if runAsUser is not set, the image
	if sc.RunAsNonRoot() == nil && sc.RunAsUser() == nil && s.scc.RunAsUser.Type == securityv1.RunAsUserStrategyMustRunAsNonRoot {
		nonRoot := true
		sc.SetRunAsNonRoot(&nonRoot)
	}

	caps, err := s.capabilitiesStrategy.Generate(pod, container)
	if err != nil {
		return nil, err
	}
	sc.SetCapabilities(caps)

	// if the SCC requires a read only root filesystem and the container has not made a specific
	// request then default ReadOnlyRootFilesystem to true.
	if s.scc.ReadOnlyRootFilesystem && sc.ReadOnlyRootFilesystem() == nil {
		readOnlyRootFS := true
		sc.SetReadOnlyRootFilesystem(&readOnlyRootFS)
	}

	// if the SCC sets DefaultAllowPrivilegeEscalation and the container security context
	// allowPrivilegeEscalation is not set, then default to that set by the SCC.
	if s.scc.DefaultAllowPrivilegeEscalation != nil && sc.AllowPrivilegeEscalation() == nil {
		sc.SetAllowPrivilegeEscalation(s.scc.DefaultAllowPrivilegeEscalation)
	}

	// if the SCC sets AllowPrivilegeEscalation to false set that as the default
	if s.scc.AllowPrivilegeEscalation != nil && !*s.scc.AllowPrivilegeEscalation && sc.AllowPrivilegeEscalation() == nil {
		sc.SetAllowPrivilegeEscalation(s.scc.AllowPrivilegeEscalation)
	}

	return sc.ContainerSecurityContext(), nil
}

// Ensure a pod's SecurityContext is in compliance with the given constraints.
func (s *simpleProvider) ValidatePodSecurityContext(pod *api.Pod, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	sc := securitycontext.NewPodSecurityContextAccessor(pod.Spec.SecurityContext)

	fsGroups := []int64{}
	if fsGroup := sc.FSGroup(); fsGroup != nil {
		fsGroups = append(fsGroups, *fsGroup)
	}
	allErrs = append(allErrs, s.fsGroupStrategy.Validate(pod, fsGroups)...)
	allErrs = append(allErrs, s.supplementalGroupStrategy.Validate(pod, sc.SupplementalGroups())...)
	allErrs = append(allErrs, s.seccompStrategy.ValidatePod(pod)...)

	allErrs = append(allErrs, s.seLinuxStrategy.Validate(fldPath.Child("seLinuxOptions"), pod, nil, sc.SELinuxOptions())...)

	if !s.scc.AllowHostNetwork && sc.HostNetwork() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), sc.HostNetwork(), "Host network is not allowed to be used"))
	}

	if !s.scc.AllowHostPID && sc.HostPID() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), sc.HostPID(), "Host PID is not allowed to be used"))
	}

	if !s.scc.AllowHostIPC && sc.HostIPC() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), sc.HostIPC(), "Host IPC is not allowed to be used"))
	}

	allErrs = append(allErrs, s.sysctlsStrategy.Validate(pod)...)

	if len(pod.Spec.Volumes) > 0 && !sccutil.SCCAllowsAllVolumes(s.scc) {
		allowedVolumes := sccutil.FSTypeToStringSetInternal(s.scc.Volumes)
		for i, v := range pod.Spec.Volumes {
			fsType, err := sccutil.GetVolumeFSType(v)
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

	if len(pod.Spec.Volumes) > 0 && len(s.scc.AllowedFlexVolumes) > 0 && sccutil.SCCAllowsFSTypeInternal(s.scc, securityv1.FSTypeFlexVolume) {
		for i, v := range pod.Spec.Volumes {
			if v.FlexVolume == nil {
				continue
			}

			found := false
			driver := v.FlexVolume.Driver
			for _, allowedFlexVolume := range s.scc.AllowedFlexVolumes {
				if driver == allowedFlexVolume.Driver {
					found = true
					break
				}
			}
			if !found {
				allErrs = append(allErrs,
					field.Invalid(fldPath.Child("volumes").Index(i).Child("driver"), driver,
						"Flexvolume driver is not allowed to be used"))
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

	allErrs = append(allErrs, s.runAsUserStrategy.Validate(fldPath.Child("securityContext"), pod, container, sc.RunAsNonRoot(), sc.RunAsUser())...)
	allErrs = append(allErrs, s.seLinuxStrategy.Validate(fldPath.Child("seLinuxOptions"), pod, container, sc.SELinuxOptions())...)
	allErrs = append(allErrs, s.seccompStrategy.ValidateContainer(pod, container)...)

	privileged := sc.Privileged()
	if !s.scc.AllowPrivilegedContainer && privileged != nil && *privileged {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("privileged"), *privileged, "Privileged containers are not allowed"))
	}

	allErrs = append(allErrs, s.capabilitiesStrategy.Validate(pod, container, sc.Capabilities())...)

	if !s.scc.AllowHostNetwork && podSC.HostNetwork() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), podSC.HostNetwork(), "Host network is not allowed to be used"))
	}

	if !s.scc.AllowHostPorts {
		containersPath := fldPath.Child("containers")
		for idx, c := range pod.Spec.Containers {
			idxPath := containersPath.Index(idx)
			allErrs = append(allErrs, s.hasHostPort(&c, idxPath)...)
		}

		containersPath = fldPath.Child("initContainers")
		for idx, c := range pod.Spec.InitContainers {
			idxPath := containersPath.Index(idx)
			allErrs = append(allErrs, s.hasHostPort(&c, idxPath)...)
		}
	}

	if !s.scc.AllowHostPID && podSC.HostPID() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), podSC.HostPID(), "Host PID is not allowed to be used"))
	}

	if !s.scc.AllowHostIPC && podSC.HostIPC() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), podSC.HostIPC(), "Host IPC is not allowed to be used"))
	}

	if s.scc.ReadOnlyRootFilesystem {
		readOnly := sc.ReadOnlyRootFilesystem()
		if readOnly == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), readOnly, "ReadOnlyRootFilesystem may not be nil and must be set to true"))
		} else if !*readOnly {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), *readOnly, "ReadOnlyRootFilesystem must be set to true"))
		}
	}

	allowEscalation := sc.AllowPrivilegeEscalation()
	if s.scc.AllowPrivilegeEscalation != nil && !*s.scc.AllowPrivilegeEscalation {
		if allowEscalation == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("allowPrivilegeEscalation"), allowEscalation, "Allowing privilege escalation for containers is not allowed"))
		}

		if allowEscalation != nil && *allowEscalation {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("allowPrivilegeEscalation"), *allowEscalation, "Allowing privilege escalation for containers is not allowed"))
		}
	}

	return allErrs
}

// hasHostPort checks the port definitions on the container for HostPort > 0.
func (s *simpleProvider) hasHostPort(container *api.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, cp := range container.Ports {
		if cp.HostPort > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPort"), cp.HostPort, "Host ports are not allowed to be used"))
		}
	}
	return allErrs
}

// Get the name of the SCC that this provider was initialized with.
func (s *simpleProvider) GetSCCName() string {
	return s.scc.Name
}

func (s *simpleProvider) GetSCCUsers() []string {
	return s.scc.Users
}

func (s *simpleProvider) GetSCCGroups() []string {
	return s.scc.Groups
}

// createUserStrategy creates a new user strategy.
func createUserStrategy(opts *securityv1.RunAsUserStrategyOptions) (user.RunAsUserSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case securityv1.RunAsUserStrategyMustRunAs:
		return user.NewMustRunAs(opts)
	case securityv1.RunAsUserStrategyMustRunAsRange:
		return user.NewMustRunAsRange(opts)
	case securityv1.RunAsUserStrategyMustRunAsNonRoot:
		return user.NewRunAsNonRoot(opts)
	case securityv1.RunAsUserStrategyRunAsAny:
		return user.NewRunAsAny(opts)
	default:
		return nil, fmt.Errorf("Unrecognized RunAsUser strategy type %s", opts.Type)
	}
}

// createSELinuxStrategy creates a new selinux strategy.
func createSELinuxStrategy(opts *securityv1.SELinuxContextStrategyOptions) (selinux.SELinuxSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case securityv1.SELinuxStrategyMustRunAs:
		return selinux.NewMustRunAs(opts)
	case securityv1.SELinuxStrategyRunAsAny:
		return selinux.NewRunAsAny(opts)
	default:
		return nil, fmt.Errorf("Unrecognized SELinuxContext strategy type %s", opts.Type)
	}
}

// createFSGroupStrategy creates a new fsgroup strategy
func createFSGroupStrategy(opts *securityv1.FSGroupStrategyOptions) (group.GroupSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case securityv1.FSGroupStrategyRunAsAny:
		return group.NewRunAsAny()
	case securityv1.FSGroupStrategyMustRunAs:
		return group.NewMustRunAs(opts.Ranges, fsGroupField)
	default:
		return nil, fmt.Errorf("Unrecognized FSGroup strategy type %s", opts.Type)
	}
}

// createSupplementalGroupStrategy creates a new supplemental group strategy
func createSupplementalGroupStrategy(opts *securityv1.SupplementalGroupsStrategyOptions) (group.GroupSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case securityv1.SupplementalGroupsStrategyRunAsAny:
		return group.NewRunAsAny()
	case securityv1.SupplementalGroupsStrategyMustRunAs:
		return group.NewMustRunAs(opts.Ranges, supplementalGroupsField)
	default:
		return nil, fmt.Errorf("Unrecognized SupplementalGroups strategy type %s", opts.Type)
	}
}

// createCapabilitiesStrategy creates a new capabilities strategy.
func createCapabilitiesStrategy(defaultAddCaps, requiredDropCaps, allowedCaps []corev1.Capability) (capabilities.CapabilitiesSecurityContextConstraintsStrategy, error) {
	return capabilities.NewDefaultCapabilities(defaultAddCaps, requiredDropCaps, allowedCaps)
}

// createSeccompStrategy creates a new seccomp strategy
func createSeccompStrategy(allowedProfiles []string) (seccomp.SeccompStrategy, error) {
	return seccomp.NewWithSeccompProfile(allowedProfiles)
}

// createSysctlsStrategy creates a new sysctls strategy
func createSysctlsStrategy(safeWhitelist, allowedUnsafeSysctls, forbiddenSysctls []string) (sysctl.SysctlsStrategy, error) {
	return sysctl.NewMustMatchPatterns(safeWhitelist, allowedUnsafeSysctls, forbiddenSysctls), nil
}
