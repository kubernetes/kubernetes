// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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
	"github.com/docker/docker/pkg/sysinfo"

	"k8s.io/api/core/v1"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// applyPlatformSpecificContainerConfig applies platform specific configurations to runtimeapi.ContainerConfig.
func (m *kubeGenericRuntimeManager) applyPlatformSpecificContainerConfig(config *runtimeapi.ContainerConfig, container *v1.Container, pod *v1.Pod, uid *int64, username string) error {
	windowsConfig, err := m.generateWindowsContainerConfig(container, pod, uid, username)
	if err != nil {
		return err
	}

	config.Windows = windowsConfig
	return nil
}

// generateWindowsContainerConfig generates windows container config for kubelet runtime v1.
// Refer https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/cri-windows.md.
func (m *kubeGenericRuntimeManager) generateWindowsContainerConfig(container *v1.Container, pod *v1.Pod, uid *int64, username string) (*runtimeapi.WindowsContainerConfig, error) {
	wc := &runtimeapi.WindowsContainerConfig{
		Resources:       &runtimeapi.WindowsContainerResources{},
		SecurityContext: &runtimeapi.WindowsContainerSecurityContext{},
	}

	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	isolatedByHyperv := kubeletapis.ShouldIsolatedByHyperV(pod.Annotations)
	if !cpuLimit.IsZero() {
		// Note that sysinfo.NumCPU() is limited to 64 CPUs on Windows due to Processor Groups,
		// as only 64 processors are available for execution by a given process. This causes
		// some oddities on systems with more than 64 processors.
		// Refer https://msdn.microsoft.com/en-us/library/windows/desktop/dd405503(v=vs.85).aspx.
		cpuMaximum := 10000 * cpuLimit.MilliValue() / int64(sysinfo.NumCPU()) / 1000
		if isolatedByHyperv {
			cpuCount := int64(cpuLimit.MilliValue()+999) / 1000
			wc.Resources.CpuCount = cpuCount

			if cpuCount != 0 {
				cpuMaximum = cpuLimit.MilliValue() / cpuCount * 10000 / 1000
			}
		}
		// ensure cpuMaximum is in range [1, 10000].
		if cpuMaximum < 1 {
			cpuMaximum = 1
		} else if cpuMaximum > 10000 {
			cpuMaximum = 10000
		}

		wc.Resources.CpuMaximum = cpuMaximum
	}

	cpuShares := milliCPUToShares(cpuLimit.MilliValue(), isolatedByHyperv)
	if cpuShares == 0 {
		cpuShares = milliCPUToShares(cpuRequest.MilliValue(), isolatedByHyperv)
	}
	wc.Resources.CpuShares = cpuShares

	memoryLimit := container.Resources.Limits.Memory().Value()
	if memoryLimit != 0 {
		wc.Resources.MemoryLimitInBytes = memoryLimit
	}

	// setup security context
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	// RunAsUser only supports int64 from Kubernetes API, but Windows containers only support username.
	if effectiveSc.RunAsUser != nil {
		return nil, fmt.Errorf("run as uid (%d) is not supported on Windows", *effectiveSc.RunAsUser)
	}
	if username != "" {
		wc.SecurityContext.RunAsUsername = username
	}

	return wc, nil
}
