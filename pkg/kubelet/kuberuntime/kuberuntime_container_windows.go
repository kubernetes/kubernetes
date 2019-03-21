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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
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

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsGMSA) {
		determineEffectiveSecurityContext(config, container, pod)
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

const (
	// GMSASpecContainerAnnotationKey is the container annotation where we store the contents of the GMSA credential spec to use.
	GMSASpecContainerAnnotationKey = "container.alpha.windows.kubernetes.io/gmsa-credential-spec"
	// gMSAContainerSpecPodAnnotationKeySuffix is the suffix of the pod annotation where the GMSA webhook admission controller
	// stores the contents of the GMSA credential spec for a given container (the full annotation being the container's name
	// with this suffix appended).
	gMSAContainerSpecPodAnnotationKeySuffix = "." + GMSASpecContainerAnnotationKey
	// gMSAPodSpecPodAnnotationKey is the pod annotation where the GMSA webhook admission controller stores the contents of the GMSA
	// credential spec to use for containers that do not have their own specific GMSA cred spec set via a
	// gMSAContainerSpecPodAnnotationKeySuffix annotation as explained above
	gMSAPodSpecPodAnnotationKey = "pod.alpha.windows.kubernetes.io/gmsa-credential-spec"
)

// determineEffectiveSecurityContext determines the effective GMSA credential spec and, if any, copies it to the container's
// GMSASpecContainerAnnotationKey annotation.
func determineEffectiveSecurityContext(config *runtimeapi.ContainerConfig, container *v1.Container, pod *v1.Pod) {
	var containerCredSpec string

	containerGMSAPodAnnotation := container.Name + gMSAContainerSpecPodAnnotationKeySuffix
	if pod.Annotations[containerGMSAPodAnnotation] != "" {
		containerCredSpec = pod.Annotations[containerGMSAPodAnnotation]
	} else if pod.Annotations[gMSAPodSpecPodAnnotationKey] != "" {
		containerCredSpec = pod.Annotations[gMSAPodSpecPodAnnotationKey]
	}

	if containerCredSpec != "" {
		if config.Annotations == nil {
			config.Annotations = make(map[string]string)
		}
		config.Annotations[GMSASpecContainerAnnotationKey] = containerCredSpec
	} else {
		// the annotation shouldn't be present, but let's err on the side of caution:
		// it should only be set here and nowhere else
		delete(config.Annotations, GMSASpecContainerAnnotationKey)
	}
}
