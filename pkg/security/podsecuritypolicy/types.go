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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/apparmor"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/capabilities"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/group"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/selinux"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/sysctl"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/user"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// Provider provides the implementation to generate a new security
// context based on constraints or validate an existing security context against constraints.
type Provider interface {
	// Create a PodSecurityContext based on the given constraints. Also returns an updated set
	// of Pod annotations for alpha feature support.
	CreatePodSecurityContext(pod *api.Pod) (*api.PodSecurityContext, map[string]string, error)
	// Create a container SecurityContext based on the given constraints. Also returns an updated set
	// of Pod annotations for alpha feature support.
	CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, map[string]string, error)
	// Ensure a pod's SecurityContext is in compliance with the given constraints.
	ValidatePodSecurityContext(pod *api.Pod, fldPath *field.Path) field.ErrorList
	// Ensure a container's SecurityContext is in compliance with the given constraints
	ValidateContainerSecurityContext(pod *api.Pod, container *api.Container, fldPath *field.Path) field.ErrorList
	// Get the name of the PSP that this provider was initialized with.
	GetPSPName() string
}

// StrategyFactory abstracts how the strategies are created from the provider so that you may
// implement your own custom strategies that may pull information from other resources as necessary.
// For example, if you would like to populate the strategies with values from namespace annotations
// you may create a factory with a client that can pull the namespace and populate the appropriate
// values.
type StrategyFactory interface {
	// CreateStrategies creates the strategies that a provider will use.  The namespace argument
	// should be the namespace of the object being checked (the pod's namespace).
	CreateStrategies(psp *extensions.PodSecurityPolicy, namespace string) (*ProviderStrategies, error)
}

// ProviderStrategies is a holder for all strategies that the provider requires to be populated.
type ProviderStrategies struct {
	RunAsUserStrategy         user.RunAsUserStrategy
	SELinuxStrategy           selinux.SELinuxStrategy
	AppArmorStrategy          apparmor.Strategy
	FSGroupStrategy           group.GroupStrategy
	SupplementalGroupStrategy group.GroupStrategy
	CapabilitiesStrategy      capabilities.Strategy
	SysctlsStrategy           sysctl.SysctlsStrategy
	SeccompStrategy           seccomp.Strategy
}
