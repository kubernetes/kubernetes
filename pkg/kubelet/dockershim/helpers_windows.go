// +build windows

/*
Copyright 2015 The Kubernetes Authors.

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

package dockershim

import (
	"os"

	"github.com/blang/semver"
	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	dockerfilters "github.com/docker/docker/api/types/filters"
	"github.com/golang/glog"

	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

func DefaultMemorySwap() int64 {
	return 0
}

func (ds *dockerService) getSecurityOpts(seccompProfile string, separator rune) ([]string, error) {
	if seccompProfile != "" {
		glog.Warningf("seccomp annotations are not supported on windows")
	}
	return nil, nil
}

// applyExperimentalCreateConfig applys experimental configures from sandbox annotations.
func applyExperimentalCreateConfig(createConfig *dockertypes.ContainerCreateConfig, annotations map[string]string) {
	if kubeletapis.ShouldIsolatedByHyperV(annotations) {
		createConfig.HostConfig.Isolation = kubeletapis.HypervIsolationValue

		if networkMode := os.Getenv("CONTAINER_NETWORK"); networkMode == "" {
			createConfig.HostConfig.NetworkMode = dockercontainer.NetworkMode("none")
		}
	}
}

func (ds *dockerService) updateCreateConfig(
	createConfig *dockertypes.ContainerCreateConfig,
	config *runtimeapi.ContainerConfig,
	sandboxConfig *runtimeapi.PodSandboxConfig,
	podSandboxID string, securityOptSep rune, apiVersion *semver.Version) error {
	if networkMode := os.Getenv("CONTAINER_NETWORK"); networkMode != "" {
		createConfig.HostConfig.NetworkMode = dockercontainer.NetworkMode(networkMode)
	} else if !kubeletapis.ShouldIsolatedByHyperV(sandboxConfig.Annotations) {
		// Todo: Refactor this call in future for calling methods directly in security_context.go
		modifyHostOptionsForContainer(nil, podSandboxID, createConfig.HostConfig)
	}

	// Apply Windows-specific options if applicable.
	if wc := config.GetWindows(); wc != nil {
		rOpts := wc.GetResources()
		if rOpts != nil {
			createConfig.HostConfig.Resources = dockercontainer.Resources{
				Memory:     rOpts.MemoryLimitInBytes,
				CPUShares:  rOpts.CpuShares,
				CPUCount:   rOpts.CpuCount,
				CPUPercent: rOpts.CpuMaximum,
			}
		}

		// Apply security context.
		applyWindowsContainerSecurityContext(wc.GetSecurityContext(), createConfig.Config, createConfig.HostConfig)
	}

	applyExperimentalCreateConfig(createConfig, sandboxConfig.Annotations)

	return nil
}

// applyWindowsContainerSecurityContext updates docker container options according to security context.
func applyWindowsContainerSecurityContext(wsc *runtimeapi.WindowsContainerSecurityContext, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	if wsc == nil {
		return
	}

	if wsc.GetRunAsUsername() != "" {
		config.User = wsc.GetRunAsUsername()
	}
}

func (ds *dockerService) determinePodIPBySandboxID(sandboxID string, sandbox *dockertypes.ContainerJSON) string {
	// Versions and feature support
	// ============================
	// Windows version >= Windows Server, Version 1709, Supports both sandbox and non-sandbox case
	// Windows version == Windows Server 2016 Support only non-sandbox case
	// Windows version < Windows Server 2016 is Not Supported

	// Sandbox support in Windows mandates CNI Plugin.
	// Presence of CONTAINER_NETWORK flag is considered as non-Sandbox cases here
	// Hyper-V isolated containers are also considered as non-Sandbox cases

	// Todo: Add a kernel version check for more validation

	// Hyper-V only supports one container per Pod yet and the container will have a different
	// IP address from sandbox. Retrieve the IP from the containers as this is a non-Sandbox case.
	// TODO(feiskyer): remove this workaround after Hyper-V supports multiple containers per Pod.
	if networkMode := os.Getenv("CONTAINER_NETWORK"); networkMode == "" && sandbox.HostConfig.Isolation != kubeletapis.HypervIsolationValue {
		// Sandbox case, fetch the IP from the sandbox container.
		return ds.getIP(sandboxID, sandbox)
	}

	// Non-Sandbox case, fetch the IP from the containers within the Pod.
	opts := dockertypes.ContainerListOptions{
		All:     true,
		Filters: dockerfilters.NewArgs(),
	}

	f := newDockerFilter(&opts.Filters)
	f.AddLabel(containerTypeLabelKey, containerTypeLabelContainer)
	f.AddLabel(sandboxIDLabelKey, sandboxID)
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return ""
	}

	for _, c := range containers {
		r, err := ds.client.InspectContainer(c.ID)
		if err != nil {
			continue
		}

		if containerIP := ds.getIP(c.ID, r); containerIP != "" {
			return containerIP
		}
	}

	return ""
}

func getNetworkNamespace(c *dockertypes.ContainerJSON) (string, error) {
	// Currently in windows there is no identifier exposed for network namespace
	// Like docker, the referenced container id is used to figure out the network namespace id internally by the platform
	// so returning the docker networkMode (which holds container:<ref containerid> for network namespace here
	return string(c.HostConfig.NetworkMode), nil
}
