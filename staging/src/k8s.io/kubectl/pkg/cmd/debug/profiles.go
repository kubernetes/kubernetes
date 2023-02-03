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
	"k8s.io/utils/pointer"
)

type legacyProfile struct {
}

type generalProfile struct {
}

type baselineProfile struct {
}

type restrictedProfile struct {
}

func (p *legacyProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	switch target.(type) {
	case *corev1.Pod:
		// do nothing to the copied pod
		return nil
	case *corev1.Node:
		MountRootPartition(pod, containerName)
		UseHostNamespaces(pod)
		return nil
	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileLegacy, target)
	}
}

type debugStyle int

const (
	// debug by ephemeral container
	styleEphemeral debugStyle = iota
	// debug by pod copy
	stylePodCopy
	// debug node
	styleNode
	// unsupported debug methodology
	styleUnsupported
)

func getDebugStyle(pod *corev1.Pod, target runtime.Object) (debugStyle, error) {
	switch target.(type) {
	case *corev1.Pod:
		if asserted, ok := target.(*corev1.Pod); ok {
			if pod != asserted { // comparing addresses
				return stylePodCopy, nil
			}
		}
		return styleEphemeral, nil
	case *corev1.Node:
		return styleNode, nil
	}
	return styleUnsupported, fmt.Errorf("objects of type %T are not supported", target)
}

func (p *generalProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("general profile: %s", err)
	}

	switch style {
	case styleNode:
		MountRootPartition(pod, containerName)
		ClearSecurityContext(pod, containerName, Containers)
		UseHostNamespaces(pod)

	case stylePodCopy:
		RemoveLabelsAndProbes(pod)
		AllowProcessTracing(pod, containerName, Containers)
		ShareProcessNamespace(pod)

	case styleEphemeral:
		AllowProcessTracing(pod, containerName, EphemeralContainers)
	}

	return nil
}

func (p *baselineProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("baseline profile: %s", err)
	}

	ClearSecurityContext(pod, containerName, Containers|EphemeralContainers)

	switch style {
	case stylePodCopy:
		RemoveLabelsAndProbes(pod)
		ShareProcessNamespace(pod)

	case styleEphemeral, styleNode:
		// no additional modifications needed
	}

	return nil
}

func (p *restrictedProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("restricted profile: %s", err)
	}

	DisallowRoot(pod, containerName, Containers|EphemeralContainers)
	DropCapabilities(pod, containerName, Containers|EphemeralContainers)

	switch style {
	case styleNode:
		ClearSecurityContext(pod, containerName, Containers)

	case stylePodCopy:
		ShareProcessNamespace(pod)

	case styleEphemeral:
		// no additional modifications needed
	}

	return nil
}

// ContainerType signifies container type
type ContainerType int

const (
	// Containers is for normal containers
	Containers ContainerType = 1 << iota
	// EphemeralContainers is for ephemeral containers
	EphemeralContainers
)

// RemoveLabelsAndProbes removes labels from the pod and remove probes
// from all containers of the pod.
func RemoveLabelsAndProbes(p *corev1.Pod) {
	p.Labels = nil
	for i := range p.Spec.Containers {
		p.Spec.Containers[i].LivenessProbe = nil
		p.Spec.Containers[i].ReadinessProbe = nil
	}
}

// MountRootPartition mounts the host's root path at "/host" in the container.
func MountRootPartition(p *corev1.Pod, containerName string) {
	const volumeName = "host-root"
	p.Spec.Volumes = append(p.Spec.Volumes, corev1.Volume{
		Name: volumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{Path: "/"},
		},
	})
	modifyContainer(p, containerName, Containers, func(c *corev1.Container) {
		c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
			MountPath: "/host",
			Name:      volumeName,
		})
	})
}

// UseHostNamespaces configures the pod to use the host's network, PID, and IPC
// namespaces.
func UseHostNamespaces(p *corev1.Pod) {
	p.Spec.HostNetwork = true
	p.Spec.HostPID = true
	p.Spec.HostIPC = true
}

// ShareProcessNamespace configures all containers in the pod to share the
// process namespace.
func ShareProcessNamespace(p *corev1.Pod) {
	p.Spec.ShareProcessNamespace = pointer.BoolPtr(true)
}

// ClearSecurityContext clears the security context for the container.
func ClearSecurityContext(p *corev1.Pod, containerName string, mask ContainerType) {
	modifyContainer(p, containerName, mask, func(c *corev1.Container) {
		c.SecurityContext = nil
	})
}

// DisallowRoot configures the container to run as a non-root user.
func DisallowRoot(p *corev1.Pod, containerName string, mask ContainerType) {
	modifyContainer(p, containerName, mask, func(c *corev1.Container) {
		c.SecurityContext = &corev1.SecurityContext{
			RunAsNonRoot: pointer.BoolPtr(true),
		}
	})
}

// DropCapabilities drops all Capabilities for the container
func DropCapabilities(p *corev1.Pod, containerName string, mask ContainerType) {
	modifyContainer(p, containerName, mask, func(c *corev1.Container) {
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Drop: []corev1.Capability{"ALL"},
				},
			}
			return
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &corev1.Capabilities{
				Drop: []corev1.Capability{"ALL"},
			}
			return
		}
		c.SecurityContext.Capabilities.Drop = append(c.SecurityContext.Capabilities.Drop, "ALL")
	})
}

// AllowProcessTracing grants the SYS_PTRACE capability to the container.
func AllowProcessTracing(p *corev1.Pod, containerName string, mask ContainerType) {
	modifyContainer(p, containerName, mask, func(c *corev1.Container) {
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Add: []corev1.Capability{"SYS_PTRACE"},
				},
			}
			return
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &corev1.Capabilities{
				Drop: []corev1.Capability{"SYS_PTRACE"},
			}
			return
		}
		c.SecurityContext.Capabilities.Drop = append(c.SecurityContext.Capabilities.Drop, "SYS_PTRACE")
	})
}

// modifyContainer performs the modifier function m against a container from the
// the input pod spec which has the name of containerName and the type satisfying
// the mask.
func modifyContainer(pod *corev1.Pod, containerName string, mask ContainerType, m func(*corev1.Container)) {
	if mask&Containers != 0 {
		for i, c := range pod.Spec.Containers {
			if c.Name != containerName {
				continue
			}
			m(&pod.Spec.Containers[i])
		}
	}
	if mask&EphemeralContainers != 0 {
		for i, c := range pod.Spec.EphemeralContainers {
			if c.Name != containerName {
				continue
			}
			m((*corev1.Container)(&pod.Spec.EphemeralContainers[i].EphemeralContainerCommon))
		}
	}
}
