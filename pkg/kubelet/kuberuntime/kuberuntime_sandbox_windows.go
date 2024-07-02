//go:build windows
// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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

package kuberuntime

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func (m *kubeGenericRuntimeManager) applySandboxResources(pod *v1.Pod, config *runtimeapi.PodSandboxConfig) error {
	return nil
}

func (m *kubeGenericRuntimeManager) addLinuxSecurityContext(lc *runtimeapi.LinuxPodSandboxConfig, pod *v1.Pod) error {
	return nil
}

// generatePodSandboxWindowsConfig generates WindowsPodSandboxConfig from v1.Pod.
// On Windows this will get called in addition to LinuxPodSandboxConfig because not all relevant fields have been added to
// WindowsPodSandboxConfig at this time.
func (m *kubeGenericRuntimeManager) generatePodSandboxWindowsConfig(pod *v1.Pod) (*runtimeapi.WindowsPodSandboxConfig, error) {
	wc := &runtimeapi.WindowsPodSandboxConfig{
		SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.WindowsHostNetwork) {
		wc.SecurityContext.NamespaceOptions = &runtimeapi.WindowsNamespaceOption{}
		if kubecontainer.IsHostNetworkPod(pod) {
			wc.SecurityContext.NamespaceOptions.Network = runtimeapi.NamespaceMode_NODE
		} else {
			wc.SecurityContext.NamespaceOptions.Network = runtimeapi.NamespaceMode_POD
		}
	}

	// If all of the containers in a pod are HostProcess containers, set the pod's HostProcess field
	// explicitly because the container runtime requires this information at sandbox creation time.
	if kubecontainer.HasWindowsHostProcessContainer(pod) {
		// At present Windows all containers in a Windows pod must be HostProcess containers
		// and HostNetwork is required to be set.
		if !kubecontainer.AllContainersAreWindowsHostProcess(pod) {
			return nil, fmt.Errorf("pod must not contain both HostProcess and non-HostProcess containers")
		}

		if !kubecontainer.IsHostNetworkPod(pod) {
			return nil, fmt.Errorf("hostNetwork is required if Pod contains HostProcess containers")
		}

		wc.SecurityContext.HostProcess = true
	}

	sc := pod.Spec.SecurityContext
	if sc == nil || sc.WindowsOptions == nil {
		return wc, nil
	}

	wo := sc.WindowsOptions
	if wo.GMSACredentialSpec != nil {
		wc.SecurityContext.CredentialSpec = *wo.GMSACredentialSpec
	}

	if wo.RunAsUserName != nil {
		wc.SecurityContext.RunAsUsername = *wo.RunAsUserName
	}

	if kubecontainer.HasWindowsHostProcessContainer(pod) {

		if wo.HostProcess != nil && !*wo.HostProcess {
			return nil, fmt.Errorf("pod must not contain any HostProcess containers if Pod's WindowsOptions.HostProcess is set to false")
		}
	}

	return wc, nil
}
