//go:build windows
// +build windows

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

package kuberuntime

import (
	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func (m *kubeGenericRuntimeManager) applySandboxResources(pod *v1.Pod, config *runtimeapi.PodSandboxConfig) error {
	return nil
}

func getPodSandboxWindowsConfig(m *kubeGenericRuntimeManager, pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig) (*runtimeapi.WindowsPodSandboxConfig, error) {
	wc, err := m.generatePodSandboxWindowsConfig(pod)
	if err != nil {
		return nil, err
	}
	return wc, err
}

func addNonWindowsRelatedContext(lc *runtimeapi.LinuxPodSandboxConfig, sc *v1.PodSecurityContext) {
}
