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
	ephemeral debugStyle = iota
	// debug by pod copy
	podCopy
	// debug node
	node
	// unsupported debug methodology
	unsupported
)

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
		MountRootPartition(pod, containerName)
		ClearSecurityContext(pod, containerName, podutils.Containers)
		UseHostNamespaces(pod)

	case podCopy:
		RemoveLabelsAndProbes(pod)
		AllowProcessTracing(pod, containerName, podutils.Containers)
		ShareProcessNamespace(pod)

	case ephemeral:
		AllowProcessTracing(pod, containerName, podutils.EphemeralContainers)
	}

	return nil
}

func (p *baselineProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	style, err := getDebugStyle(pod, target)
	if err != nil {
		return fmt.Errorf("baseline profile: %s", err)
	}

	ClearSecurityContext(pod, containerName, podutils.Containers|podutils.EphemeralContainers)

	switch style {
	case podCopy:
		RemoveLabelsAndProbes(pod)
		ShareProcessNamespace(pod)

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

	DisallowRoot(pod, containerName, podutils.Containers|podutils.EphemeralContainers)
	DropCapabilities(pod, containerName, podutils.Containers|podutils.EphemeralContainers)

	switch style {
	case node:
		ClearSecurityContext(pod, containerName, podutils.Containers)

	case podCopy:
		ShareProcessNamespace(pod)

	case ephemeral:
		// no additional modifications needed
	}

	return nil
}

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
func ClearSecurityContext(p *corev1.Pod, containerName string, mask podutils.ContainerType) {
	podutils.VisitContainers(&p.Spec, mask, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		c.SecurityContext = nil
		return false
	})
}

// DisallowRoot configures the container to run as a non-root user.
func DisallowRoot(p *corev1.Pod, containerName string, mask podutils.ContainerType) {
	podutils.VisitContainers(&p.Spec, mask, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		c.SecurityContext = &corev1.SecurityContext{
			RunAsNonRoot: pointer.BoolPtr(true),
		}
		return false
	})
}

// DropCapabilities drops all Capabilities for the container
func DropCapabilities(p *corev1.Pod, containerName string, mask podutils.ContainerType) {
	podutils.VisitContainers(&p.Spec, mask, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Drop: []corev1.Capability{"ALL"},
				},
			}
			return false
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &corev1.Capabilities{
				Drop: []corev1.Capability{"ALL"},
			}
			return false
		}
		c.SecurityContext.Capabilities.Drop = append(c.SecurityContext.Capabilities.Drop, "ALL")
		return false
	})
}

// AllowProcessTracing grants the SYS_PTRACE capability to the container.
func AllowProcessTracing(p *corev1.Pod, containerName string, mask podutils.ContainerType) {
	podutils.VisitContainers(&p.Spec, mask, func(c *corev1.Container, _ podutils.ContainerType) bool {
		if c.Name != containerName {
			return true
		}
		if c.SecurityContext == nil {
			c.SecurityContext = &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Add: []corev1.Capability{"SYS_PTRACE"},
				},
			}
			return false
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &corev1.Capabilities{
				Drop: []corev1.Capability{"SYS_PTRACE"},
			}
			return false
		}
		c.SecurityContext.Capabilities.Drop = append(c.SecurityContext.Capabilities.Drop, "SYS_PTRACE")
		return false
	})
}
