package sccmatching

import (
	securityv1 "github.com/openshift/api/security/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// SecurityContextConstraintsProvider provides the implementation to generate a new security
// context based on constraints or validate an existing security context against constraints.
type SecurityContextConstraintsProvider interface {
	// Create a PodSecurityContext based on the given constraints.
	CreatePodSecurityContext(pod *api.Pod) (*api.PodSecurityContext, map[string]string, error)
	// Create a container SecurityContext based on the given constraints
	CreateContainerSecurityContext(pod *api.Pod, container *api.Container) (*api.SecurityContext, error)
	// Ensure a pod's SecurityContext is in compliance with the given constraints.
	ValidatePodSecurityContext(pod *api.Pod, fldPath *field.Path) field.ErrorList
	// Ensure a container's SecurityContext is in compliance with the given constraints
	ValidateContainerSecurityContext(pod *api.Pod, container *api.Container, fldPath *field.Path) field.ErrorList
	// Get the SCC that this provider was initialized with.
	GetSCC() *securityv1.SecurityContextConstraints
	// Get the name of the SCC that this provider was initialized with.
	GetSCCName() string
	// Get the users associated to the SCC this provider was initialized with
	GetSCCUsers() []string
	// Get the groups associated to the SCC this provider was initialized with
	GetSCCGroups() []string
}
