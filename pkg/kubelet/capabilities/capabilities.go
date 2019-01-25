/*
Copyright 2019 The Kubernetes Authors.

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

package capabilities

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	capabilitiespkg "k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/securitycontext"
)

const (
	// ForbiddenReason this will be added in the pod status
	ForbiddenReason = "CapabilitiesForbidden"
)

// Capabilities type allows you to check the capabilities in pod and the ones
// allowed on the node, and acts as a soft admit handler for kubelet before
// running it on the node
type Capabilities struct {
	capabilitiespkg.Capabilities
}

// NewCapabilities creates a capabilities object based on the Capabilities
// object defined in package `k8s.io/kubernetes/pkg/capabilities`
func NewCapabilities(c capabilitiespkg.Capabilities) *Capabilities {
	return &Capabilities{c}
}

func rejectPod(msg string) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{
		Admit:   false,
		Reason:  ForbiddenReason,
		Message: msg,
	}
}

// Admit satisfies the interface to check if capabilities allowed and the ones
// passed in the pod satisfy the requirement if it does then the pod is admitted
// otherwise rejected
func (c *Capabilities) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	pod := attrs.Pod
	if !c.AllowPrivileged {
		for _, container := range pod.Spec.Containers {
			if securitycontext.HasPrivilegedRequest(&container) {
				return rejectPod(fmt.Sprintf("pod with UID %q specified privileged container, but is disallowed", pod.UID))
			}
		}
		for _, container := range pod.Spec.InitContainers {
			if securitycontext.HasPrivilegedRequest(&container) {
				return rejectPod(fmt.Sprintf("pod with UID %q specified privileged init container, but is disallowed", pod.UID))
			}
		}
	}

	if pod.Spec.HostNetwork {
		allowed, err := c.allowHostNetwork(pod)
		if err != nil || !allowed {
			return rejectPod(fmt.Sprintf("pod with UID %q specified host networking, but is disallowed", pod.UID))
		}
	}

	if pod.Spec.HostPID {
		allowed, err := c.allowHostPID(pod)
		if err != nil || !allowed {
			return rejectPod(fmt.Sprintf("pod with UID %q specified host PID, but is disallowed", pod.UID))
		}
	}

	if pod.Spec.HostIPC {
		allowed, err := c.allowHostIPC(pod)
		if err != nil || !allowed {
			return rejectPod(fmt.Sprintf("pod with UID %q specified host IPC, but is disallowed", pod.UID))
		}
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

// Determined whether the specified pod is allowed to use host networking
func (c *Capabilities) allowHostNetwork(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range c.PrivilegedSources.HostNetworkSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}

// Determined whether the specified pod is allowed to use host PID
func (c *Capabilities) allowHostPID(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range c.PrivilegedSources.HostPIDSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}

// Determined whether the specified pod is allowed to use host ipc
func (c *Capabilities) allowHostIPC(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range c.PrivilegedSources.HostIPCSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}
