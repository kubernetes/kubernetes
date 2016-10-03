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

package securitycontextconstraints

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/securitycontextconstraints/capabilities"
	"k8s.io/kubernetes/pkg/securitycontextconstraints/group"
	"k8s.io/kubernetes/pkg/securitycontextconstraints/seccomp"
	"k8s.io/kubernetes/pkg/securitycontextconstraints/selinux"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/sysctl"
	"k8s.io/kubernetes/pkg/securitycontextconstraints/user"
	sccutil "k8s.io/kubernetes/pkg/securitycontextconstraints/util"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// used to pass in the field being validated for reusable group strategies so they
// can create informative error messages.
const (
	fsGroupField            = "fsGroup"
	supplementalGroupsField = "supplementalGroups"
)

// simpleProvider is the default implementation of SecurityContextConstraintsProvider
type simpleProvider struct {
	scc                       *api.SecurityContextConstraints
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
func NewSimpleProvider(scc *api.SecurityContextConstraints) (SecurityContextConstraintsProvider, error) {
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

	var unsafeSysctls []string
	if ann, found := scc.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey]; found {
		var err error
		unsafeSysctls, err = extensions.SysctlsFromPodSecurityPolicyAnnotation(ann)
		if err != nil {
			return nil, err
		}
	}
	sysctlsStrat, err := createSysctlsStrategy(unsafeSysctls)
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

	var annotationsCopy map[string]string = nil
	if pod.Annotations != nil {
		annotationsCopy = make(map[string]string, len(pod.Annotations))
		for k, v := range pod.Annotations {
			annotationsCopy[k] = v
		}
	}

	if len(sc.SupplementalGroups) == 0 {
		supGroups, err := s.supplementalGroupStrategy.Generate(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.SupplementalGroups = supGroups
	}

	if sc.FSGroup == nil {
		fsGroup, err := s.fsGroupStrategy.GenerateSingle(pod)
		if err != nil {
			return nil, nil, err
		}
		sc.FSGroup = fsGroup
	}

	if sc.SELinuxOptions == nil {
		seLinux, err := s.seLinuxStrategy.Generate(pod, nil)
		if err != nil {
			return nil, nil, err
		}
		sc.SELinuxOptions = seLinux
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

	return sc, annotationsCopy, nil
}

// Create a SecurityContext based on the given constraints.  If a setting is already set on the
// container's security context then it will not be changed.  Validation should be used after
// the context is created to ensure it complies with the required restrictions.
//
// NOTE: this method works on a copy of the SC of the container.  It is up to the caller to apply
// the SC if validation passes.
func (s *simpleProvider) CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, error) {
	var sc *api.SecurityContext = nil
	if container.SecurityContext != nil {
		// work with a copy of the original
		copy := *container.SecurityContext
		sc = &copy
	} else {
		sc = &api.SecurityContext{}
	}
	if sc.RunAsUser == nil {
		uid, err := s.runAsUserStrategy.Generate(pod, container)
		if err != nil {
			return nil, err
		}
		sc.RunAsUser = uid
	}

	if sc.SELinuxOptions == nil {
		seLinux, err := s.seLinuxStrategy.Generate(pod, container)
		if err != nil {
			return nil, err
		}
		sc.SELinuxOptions = seLinux
	}

	if sc.Privileged == nil {
		priv := false
		sc.Privileged = &priv
	}

	// if we're using the non-root strategy set the marker that this container should not be
	// run as root which will signal to the kubelet to do a final check either on the runAsUser
	// or, if runAsUser is not set, the image
	if s.scc.RunAsUser.Type == api.RunAsUserStrategyMustRunAsNonRoot {
		nonRoot := true
		sc.RunAsNonRoot = &nonRoot
	}

	caps, err := s.capabilitiesStrategy.Generate(pod, container)
	if err != nil {
		return nil, err
	}
	sc.Capabilities = caps

	// if the SCC requires a read only root filesystem and the container has not made a specific
	// request then default ReadOnlyRootFilesystem to true.
	if s.scc.ReadOnlyRootFilesystem && sc.ReadOnlyRootFilesystem == nil {
		readOnlyRootFS := true
		sc.ReadOnlyRootFilesystem = &readOnlyRootFS
	}

	return sc, nil
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
	allErrs = append(allErrs, s.fsGroupStrategy.Validate(pod, fsGroups)...)
	allErrs = append(allErrs, s.supplementalGroupStrategy.Validate(pod, pod.Spec.SecurityContext.SupplementalGroups)...)
	allErrs = append(allErrs, s.seccompStrategy.ValidatePod(pod)...)

	// make a dummy container context to reuse the selinux strategies
	container := &api.Container{
		Name: pod.Name,
		SecurityContext: &api.SecurityContext{
			SELinuxOptions: pod.Spec.SecurityContext.SELinuxOptions,
		},
	}
	allErrs = append(allErrs, s.seLinuxStrategy.Validate(pod, container)...)

	if !s.scc.AllowHostNetwork && pod.Spec.SecurityContext.HostNetwork {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), pod.Spec.SecurityContext.HostNetwork, "Host network is not allowed to be used"))
	}

	if !s.scc.AllowHostPID && pod.Spec.SecurityContext.HostPID {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), pod.Spec.SecurityContext.HostPID, "Host PID is not allowed to be used"))
	}

	if !s.scc.AllowHostIPC && pod.Spec.SecurityContext.HostIPC {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), pod.Spec.SecurityContext.HostIPC, "Host IPC is not allowed to be used"))
	}

	allErrs = append(allErrs, s.sysctlsStrategy.Validate(pod)...)

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
	allErrs = append(allErrs, s.runAsUserStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.seLinuxStrategy.Validate(pod, container)...)
	allErrs = append(allErrs, s.seccompStrategy.ValidateContainer(pod, container)...)

	if !s.scc.AllowPrivilegedContainer && *sc.Privileged {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("privileged"), *sc.Privileged, "Privileged containers are not allowed"))
	}

	allErrs = append(allErrs, s.capabilitiesStrategy.Validate(pod, container)...)

	if len(pod.Spec.Volumes) > 0 && !sccutil.SCCAllowsAllVolumes(s.scc) {
		allowedVolumes := sccutil.FSTypeToStringSet(s.scc.Volumes)
		for i, v := range pod.Spec.Volumes {
			fsType, err := sccutil.GetVolumeFSType(v)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("volumes").Index(i), string(fsType), err.Error()))
				continue
			}

			if !allowedVolumes.Has(string(fsType)) {
				allErrs = append(allErrs, field.Invalid(
					fldPath.Child("volumes").Index(i), string(fsType),
					fmt.Sprintf("%s volumes are not allowed to be used", string(fsType))))
			}
		}
	}

	if !s.scc.AllowHostNetwork && pod.Spec.SecurityContext.HostNetwork {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostNetwork"), pod.Spec.SecurityContext.HostNetwork, "Host network is not allowed to be used"))
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

	if !s.scc.AllowHostPID && pod.Spec.SecurityContext.HostPID {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostPID"), pod.Spec.SecurityContext.HostPID, "Host PID is not allowed to be used"))
	}

	if !s.scc.AllowHostIPC && pod.Spec.SecurityContext.HostIPC {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("hostIPC"), pod.Spec.SecurityContext.HostIPC, "Host IPC is not allowed to be used"))
	}

	if s.scc.ReadOnlyRootFilesystem {
		if sc.ReadOnlyRootFilesystem == nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), sc.ReadOnlyRootFilesystem, "ReadOnlyRootFilesystem may not be nil and must be set to true"))
		} else if !*sc.ReadOnlyRootFilesystem {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("readOnlyRootFilesystem"), *sc.ReadOnlyRootFilesystem, "ReadOnlyRootFilesystem must be set to true"))
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

// createUserStrategy creates a new user strategy.
func createUserStrategy(opts *api.RunAsUserStrategyOptions) (user.RunAsUserSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case api.RunAsUserStrategyMustRunAs:
		return user.NewMustRunAs(opts)
	case api.RunAsUserStrategyMustRunAsRange:
		return user.NewMustRunAsRange(opts)
	case api.RunAsUserStrategyMustRunAsNonRoot:
		return user.NewRunAsNonRoot(opts)
	case api.RunAsUserStrategyRunAsAny:
		return user.NewRunAsAny(opts)
	default:
		return nil, fmt.Errorf("Unrecognized RunAsUser strategy type %s", opts.Type)
	}
}

// createSELinuxStrategy creates a new selinux strategy.
func createSELinuxStrategy(opts *api.SELinuxContextStrategyOptions) (selinux.SELinuxSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case api.SELinuxStrategyMustRunAs:
		return selinux.NewMustRunAs(opts)
	case api.SELinuxStrategyRunAsAny:
		return selinux.NewRunAsAny(opts)
	default:
		return nil, fmt.Errorf("Unrecognized SELinuxContext strategy type %s", opts.Type)
	}
}

// createFSGroupStrategy creates a new fsgroup strategy
func createFSGroupStrategy(opts *api.FSGroupStrategyOptions) (group.GroupSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case api.FSGroupStrategyRunAsAny:
		return group.NewRunAsAny()
	case api.FSGroupStrategyMustRunAs:
		return group.NewMustRunAs(opts.Ranges, fsGroupField)
	default:
		return nil, fmt.Errorf("Unrecognized FSGroup strategy type %s", opts.Type)
	}
}

// createSupplementalGroupStrategy creates a new supplemental group strategy
func createSupplementalGroupStrategy(opts *api.SupplementalGroupsStrategyOptions) (group.GroupSecurityContextConstraintsStrategy, error) {
	switch opts.Type {
	case api.SupplementalGroupsStrategyRunAsAny:
		return group.NewRunAsAny()
	case api.SupplementalGroupsStrategyMustRunAs:
		return group.NewMustRunAs(opts.Ranges, supplementalGroupsField)
	default:
		return nil, fmt.Errorf("Unrecognized SupplementalGroups strategy type %s", opts.Type)
	}
}

// createCapabilitiesStrategy creates a new capabilities strategy.
func createCapabilitiesStrategy(defaultAddCaps, requiredDropCaps, allowedCaps []api.Capability) (capabilities.CapabilitiesSecurityContextConstraintsStrategy, error) {
	return capabilities.NewDefaultCapabilities(defaultAddCaps, requiredDropCaps, allowedCaps)
}

// createSeccompStrategy creates a new seccomp strategy
func createSeccompStrategy(allowedProfiles []string) (seccomp.SeccompStrategy, error) {
	return seccomp.NewWithSeccompProfile(allowedProfiles)
}

// createSysctlsStrategy creates a new unsafe sysctls strategy.
func createSysctlsStrategy(sysctlsPatterns []string) (sysctl.SysctlsStrategy, error) {
	return sysctl.NewMustMatchPatterns(sysctlsPatterns)
}
