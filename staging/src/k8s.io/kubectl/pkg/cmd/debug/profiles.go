/*
Copyright 2022 The Kubernetes Authors.

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

package debug

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubectl/pkg/util/podutils"
	"k8s.io/utils/pointer"
)

type debugStyle int

const (
	// debug by ephemeral container
	ephemeral debugStyle = iota
	// debug by pod copy
	podCopy
	// debug node
	node
	// unsupported debug methodology
	unsupported
)

const (
	// NOTE: when you add a new profile string, remember to add it to the
	// --profile flag's help text

	// ProfileLegacy represents the legacy debugging profile which is backwards-compatible with 1.23 behavior.
	ProfileLegacy = "legacy"
	// ProfileGeneral contains a reasonable set of defaults tailored for each debugging journey.
	ProfileGeneral = "general"
	// ProfileBaseline is identical to "general" but eliminates privileges that are disallowed under
	// the baseline security profile, such as host namespaces, host volume, mounts and SYS_PTRACE.
	ProfileBaseline = "baseline"
	// ProfileRestricted is identical to "baseline" but adds configuration that's required
	// under the restricted security profile, such as requiring a non-root user and dropping all capabilities.
	ProfileRestricted = "restricted"
	// ProfileNetadmin offers elevated privileges for network debugging.
	ProfileNetadmin = "netadmin"
	// ProfileSysadmin offers elevated privileges for debugging.
	ProfileSysadmin = "sysadmin"
)

type ProfileApplier interface {
	// Apply applies the profile to the given container in the pod.
	Apply(pod *corev1.Pod, containerName string, target runtime.Object) error
}

// NewProfileApplier returns a new Options for the given profile name.
func NewProfileApplier(profile string) (ProfileApplier, error) {
	switch profile {
	case ProfileLegacy:
		return &legacyProfile{}, nil
	case ProfileGeneral:
		return &generalProfile{}, nil
	case ProfileBaseline:
		return &baselineProfile{}, nil
	case ProfileRestricted:
		return &restrictedProfile{}, nil
	case ProfileNetadmin:
		return &netadminProfile{}, nil
	case ProfileSysadmin:
		return &sysadminProfile{}, nil
	}

	return nil, fmt.Errorf("unknown profile: %s", profile)
}

type legacyProfile struct {
}

type generalProfile struct {
}

type baselineProfile struct {
}

type restrictedProfile struct {
}

type netadminProfile struct {
}

type sysadminProfile struct {
}

func (p *legacyProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	switch target.(type) {
	case *corev1.Pod:
		// do nothing to the copied pod
		return nil
	case *corev1.Node:
		mountRootPartition(pod, containerName)
		useHostNamespaces(pod)
		return nil
	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileLegacy, target)
	}
}

func getDebugStyle(pod *corev1.Pod, target runtime.Object) (debugStyle, error) {
	switch target.(type) {
	case *corev1.Pod:
		if asserted, ok := target.(*corev1.Pod); ok {
			if pod != asserted { // comparing addresses
				return podCopy, nil
			}
		}
		return ephemeral, nil
	case *corev1.Node:
		return node, nil
	}
	return unsupported, fmt.Errorf("objects of type %T are not supported", target)
}

func (p *generalProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("general profile: %s", err)
	}

	switch style {
	case node:
		mountRootPartition(pod, containerName)
		clearSecurityContext(pod, containerName)
		useHostNamespaces(pod)

	case podCopy:
		removeLabelsAndProbes(pod)
		allowProcessTracing(pod, containerName)
		shareProcessNamespace(pod)

	case ephemeral:
		allowProcessTracing(pod, containerName)
	}

	return nil
}

func (p *baselineProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("baseline profile: %s", err)
	}

	clearSecurityContext(pod, containerName)

	switch style {
	case podCopy:
		removeLabelsAndProbes(pod)
		shareProcessNamespace(pod)

	case ephemeral, node:
		// no additional modifications needed
	}

	return nil
}

func (p *restrictedProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("restricted profile: %s", err)
	}

	clearSecurityContext(pod, containerName)
	disallowRoot(pod, containerName)
	dropCapabilities(pod, containerName)
	disallowPrivilegeEscalation(pod, containerName)
	setSeccompProfile(pod, containerName)

	switch style {
	case podCopy:
		shareProcessNamespace(pod)

	case ephemeral, node:
		// no additional modifications needed
	}

	return nil
}

func (p *netadminProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("netadmin profile: %s", err)
	}

	allowNetadminCapability(pod, containerName)

	switch style {
	case node:
		useHostNamespaces(pod)

	case podCopy:
		shareProcessNamespace(pod)

	case ephemeral:
		// no additional modifications needed
	}

	return nil
}

func (p *sysadminProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("sysadmin profile: %s", err)
	}

	setPrivileged(pod, containerName)

	switch style {
	case node:
		useHostNamespaces(pod)
		mountRootPartition(pod, containerName)

	case podCopy:
		// to mimic general, default and baseline
		shareProcessNamespace(pod)
	case ephemeral:
		// no additional modifications needed
	}

	return nil
}

// removeLabelsAndProbes removes labels from the pod and remove probes
// from all containers of the pod.
func removeLabelsAndProbes(p *corev1.Pod) {
	p.Labels = nil
	for i := range p.Spec.Containers {
		p.Spec.Containers[i].LivenessProbe = nil
		p.Spec.Containers[i].ReadinessProbe = nil
		p.Spec.Containers[i].StartupProbe = nil
	}
}

// mountRootPartition mounts the host's root path at "/host" in the container.
func mountRootPartition(p *corev1.Pod, containerName string) {
	const volumeName = "host-root"
	p.Spec.Volumes = append(p.Spec.Volumes, corev1.Volume{
		Name: volumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{Path: "/"},
		},
	})
	podutils.VisitContainers(&p.Spec, podutils.Containers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
			MountPath: "/host",
			Name:      volumeName,
		})
		return false
	})
}

// useHostNamespaces configures the pod to use the host's network, PID, and IPC
// namespaces.
func useHostNamespaces(p *corev1.Pod) {
	p.Spec.HostNetwork = true
	p.Spec.HostPID = true
	p.Spec.HostIPC = true
}

// shareProcessNamespace configures all containers in the pod to share the
// process namespace.
func shareProcessNamespace(p *corev1.Pod) {
	if p.Spec.ShareProcessNamespace == nil {
		p.Spec.ShareProcessNamespace = pointer.Bool(true)
	}
}

// clearSecurityContext clears the security context for the container.
func clearSecurityContext(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		c.SecurityContext = nil
		return false
	})
}

// setPrivileged configures the containers as privileged.
func setPrivileged(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{}
		}
		c.SecurityContext.Privileged = pointer.Bool(true)
		return false
	})
}

// disallowRoot configures the container to run as a non-root user.
func disallowRoot(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{}
		}
		c.SecurityContext.RunAsNonRoot = pointer.Bool(true)
		return false
	})
}

// dropCapabilities drops all Capabilities for the container
func dropCapabilities(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{}
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &corev1.Capabilities{}
		}
		c.SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
		c.SecurityContext.Capabilities.Add = nil
		return false
	})
}

// allowProcessTracing grants the SYS_PTRACE capability to the container.
func allowProcessTracing(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		addCapability(c, "SYS_PTRACE")
		return false
	})
}

// allowNetadminCapability grants NET_ADMIN and NET_RAW capability to the container.
func allowNetadminCapability(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		addCapability(c, "NET_ADMIN")
		addCapability(c, "NET_RAW")
		return false
	})
}

func addCapability(c *corev1.Container, capability corev1.Capability) {
	if c.SecurityContext == nil {
		c.SecurityContext = &corev1.SecurityContext{}
	}
	if c.SecurityContext.Capabilities == nil {
		c.SecurityContext.Capabilities = &corev1.Capabilities{}
	}
	c.SecurityContext.Capabilities.Add = append(c.SecurityContext.Capabilities.Add, capability)
}

// disallowPrivilegeEscalation configures the containers not allowed PrivilegeEscalation
func disallowPrivilegeEscalation(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{}
		}
		c.SecurityContext.AllowPrivilegeEscalation = pointer.Bool(false)
		return false
	})
}

// setSeccompProfile apply SeccompProfile to the containers
func setSeccompProfile(p *corev1.Pod, containerName string) {
	podutils.VisitContainers(&p.Spec, podutils.AllContainers, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{}
		}
		c.SecurityContext.SeccompProfile = &corev1.SeccompProfile{Type: "RuntimeDefault"}
		return false
	})
}
