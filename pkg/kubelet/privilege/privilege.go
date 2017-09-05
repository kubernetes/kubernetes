/*
Copyright 2017 The Kubernetes Authors.

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

package privilege

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/securitycontext"
)

type privilegeAdmitHandler struct{}

// NewPrivilegeAdmitHandler returns an AdmissionFailureHandler
// that handles admission failure for privilege related requests of Pods.
func NewPrivilegeAdmitHandler() lifecycle.PodAdmitHandler {
	return &privilegeAdmitHandler{}
}

// Admit checks whether the privilege related requests satisfied
func (w *privilegeAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if err := canRunPod(attrs.Pod); err != nil {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  "Forbidden",
			Message: err.Error(),
		}
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

// Check whether we have the capabilities to run the specified pod.
func canRunPod(pod *v1.Pod) error {
	if !capabilities.Get().AllowPrivileged {
		for _, container := range pod.Spec.Containers {
			if securitycontext.HasPrivilegedRequest(&container) {
				return fmt.Errorf("pod with UID %q specified privileged container, but is disallowed", pod.UID)
			}
		}
		for _, container := range pod.Spec.InitContainers {
			if securitycontext.HasPrivilegedRequest(&container) {
				return fmt.Errorf("pod with UID %q specified privileged init container, but is disallowed", pod.UID)
			}
		}
	}

	if pod.Spec.HostNetwork {
		allowed, err := allowHostNetwork(pod)
		if err != nil {
			return err
		}
		if !allowed {
			return fmt.Errorf("pod with UID %q specified host networking, but is disallowed", pod.UID)
		}
	}

	if pod.Spec.HostPID {
		allowed, err := allowHostPID(pod)
		if err != nil {
			return err
		}
		if !allowed {
			return fmt.Errorf("pod with UID %q specified host PID, but is disallowed", pod.UID)
		}
	}

	if pod.Spec.HostIPC {
		allowed, err := allowHostIPC(pod)
		if err != nil {
			return err
		}
		if !allowed {
			return fmt.Errorf("pod with UID %q specified host ipc, but is disallowed", pod.UID)
		}
	}

	return nil
}

// Determined whether the specified pod is allowed to use host networking
func allowHostNetwork(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range capabilities.Get().PrivilegedSources.HostNetworkSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}

// Determined whether the specified pod is allowed to use host PID
func allowHostPID(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range capabilities.Get().PrivilegedSources.HostPIDSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}

// Determined whether the specified pod is allowed to use host ipc
func allowHostIPC(pod *v1.Pod) (bool, error) {
	podSource, err := kubetypes.GetPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range capabilities.Get().PrivilegedSources.HostIPCSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}
