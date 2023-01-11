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
		const volumeName = "host-root"
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: volumeName,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: "/"},
			},
		})

		for i := range pod.Spec.Containers {
			container := &pod.Spec.Containers[i]
			if container.Name != containerName {
				continue
			}
			container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
				MountPath: "/host",
				Name:      volumeName,
			})
		}

		pod.Spec.HostNetwork = true
		pod.Spec.HostPID = true
		pod.Spec.HostIPC = true
		return nil
	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileLegacy, target)
	}
}

func (p *generalProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	switch t := target.(type) {
	case *corev1.Pod:
		if t == pod {
			// For ephemeral container: sets SYS_PTRACE in ephemeral container
			modifyEphemeralContainer(pod.Spec.EphemeralContainers, containerName, func(c *corev1.EphemeralContainer) {
				c.SecurityContext = addCapability(c.SecurityContext, "SYS_PTRACE")
			})
		} else {
			// For copy of pod: sets SYS_PTRACE in debugging container, sets shareProcessNamespace
			pod.Spec.ShareProcessNamespace = pointer.BoolPtr(true)
			modifyContainer(pod.Spec.Containers, containerName, func(c *corev1.Container) {
				c.SecurityContext = addCapability(c.SecurityContext, "SYS_PTRACE")
			})
			removeProbes(pod.Spec.Containers)
		}
	case *corev1.Node:
		// empty securityContext; uses host namespaces, mounts root partition
		const volumeName = "host-root"
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: volumeName,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: "/"},
			},
		})
		modifyContainer(pod.Spec.Containers, containerName, func(c *corev1.Container) {
			c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
				MountPath: "/host",
				Name:      volumeName,
			})
		})
		pod.Spec.SecurityContext = nil
		pod.Spec.HostNetwork = true
		pod.Spec.HostPID = true
		pod.Spec.HostIPC = true
	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileGeneral, target)
	}

	return nil
}

func (p *baselineProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	switch t := target.(type) {
	case *corev1.Pod:
		if t == pod {
			// For ephemeral container: empty securityContext
			modifyEphemeralContainer(pod.Spec.EphemeralContainers, containerName, func(c *corev1.EphemeralContainer) {
				c.SecurityContext = nil
			})
		} else {
			// For copy of pod: empty securityContext; sets shareProcessNamespace
			pod.Spec.SecurityContext = nil
			pod.Spec.ShareProcessNamespace = pointer.BoolPtr(true)
			pod.Spec.HostNetwork = false
			pod.Spec.HostPID = false
			pod.Spec.HostIPC = false
			modifyContainer(pod.Spec.Containers, containerName, func(c *corev1.Container) {
				c.SecurityContext = nil
				c.SecurityContext = addCapability(c.SecurityContext, "SYS_PTRACE")
			})
			removeProbes(pod.Spec.Containers)
		}
	case *corev1.Node:
		// empty securityContext; uses isolated namespaces
		pod.Spec.SecurityContext = nil
		pod.Spec.HostNetwork = false
		pod.Spec.HostPID = false
		pod.Spec.HostIPC = false

	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileBaseline, target)
	}

	return nil
}

func (p *restrictedProfile) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	// drops privilege and run as non-root
	sc := &corev1.SecurityContext{
		RunAsNonRoot: pointer.BoolPtr(true),
		Capabilities: &corev1.Capabilities{
			Drop: []corev1.Capability{"ALL"},
		},
	}

	switch t := target.(type) {
	case *corev1.Pod:
		if t == pod {
			modifyEphemeralContainer(pod.Spec.EphemeralContainers, containerName, func(c *corev1.EphemeralContainer) {
				c.SecurityContext = sc
			})
		} else {
			// For copy of pod: empty securityContext; sets shareProcessNamespace
			pod.Spec.ShareProcessNamespace = pointer.BoolPtr(true)
			modifyContainer(pod.Spec.Containers, containerName, func(c *corev1.Container) {
				c.SecurityContext = sc
			})
			removeProbes(pod.Spec.Containers)
		}
	case *corev1.Node:
		// no additional settings required other than the common
		// settings applied below
	default:
		return fmt.Errorf("the %s profile doesn't support objects of type %T", ProfileRestricted, target)
	}

	// common settings, empty securityContext and uses isolated namespaces
	pod.Spec.SecurityContext = nil
	pod.Spec.HostNetwork = false
	pod.Spec.HostPID = false
	pod.Spec.HostIPC = false

	return nil
}

func addCapability(s *corev1.SecurityContext, c corev1.Capability) *corev1.SecurityContext {
	if s == nil {
		return &corev1.SecurityContext{
			Capabilities: &corev1.Capabilities{
				Add: []corev1.Capability{c},
			},
		}
	}

	if s.Capabilities == nil {
		s.Capabilities = &corev1.Capabilities{
			Add: []corev1.Capability{c},
		}
		return s
	}

	s.Capabilities.Add = append(s.Capabilities.Add, c)
	return s
}

// removeProbes remove liveness and readiness probes from the supplied list of containers
func removeProbes(cs []corev1.Container) {
	for i := range cs {
		cs[i].LivenessProbe = nil
		cs[i].ReadinessProbe = nil
	}
}

// modifyContainer performs m against a container from cs which has the name of containerName.
func modifyContainer(cs []corev1.Container, containerName string, m func(*corev1.Container)) {
	for i, c := range cs {
		if c.Name != containerName {
			continue
		}
		m(&cs[i])
	}
}

// modifyEphemeralContainer performs m against a container from cs which has the name of containerName.
func modifyEphemeralContainer(cs []corev1.EphemeralContainer, containerName string, m func(*corev1.EphemeralContainer)) {
	for i, c := range cs {
		if c.Name != containerName {
			continue
		}
		m(&cs[i])
	}
}
