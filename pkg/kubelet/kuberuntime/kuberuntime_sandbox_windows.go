// +build windows

/*
Copyright 2020 The Kubernetes Authors.

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
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

// applyPodSandboxWindowsConfig generates WindowsPodSandboxConfig from v1.Pod.
func (m *kubeGenericRuntimeManager) applyPodSandboxConfig(pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig) error {

	wc, err := m.generatePodSandboxWindowsConfig(pod)
	if err != nil {
		return err
	}
	podSandboxConfig.Windows = wc
	return nil
}

func (m *kubeGenericRuntimeManager) generatePodSandboxWindowsConfig(pod *v1.Pod) (*runtimeapi.WindowsPodSandboxConfig, error) {
	wc := &runtimeapi.WindowsPodSandboxConfig{
		SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
	}
	if pod.Spec.SecurityContext != nil {
		sc := pod.Spec.SecurityContext
		if sc != nil {
			wsc := pod.Spec.SecurityContext.WindowsOptions
			if wsc.RunAsUserName != nil {
				wc.SecurityContext.RunAsUser = *wsc.RunAsUserName
			}
		}
	}
	return wc, nil
}
