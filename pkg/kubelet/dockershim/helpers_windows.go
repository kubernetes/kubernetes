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
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
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

func (ds *dockerService) updateCreateConfig(
	createConfig *dockertypes.ContainerCreateConfig,
	config *runtimeapi.ContainerConfig,
	sandboxConfig *runtimeapi.PodSandboxConfig,
	podSandboxID string, securityOptSep rune, apiVersion *semver.Version) error {
	if networkMode := os.Getenv("CONTAINER_NETWORK"); networkMode != "" {
		createConfig.HostConfig.NetworkMode = dockercontainer.NetworkMode(networkMode)
	} else {
		// Todo: Refactor this call in future for calling methods directly in security_context.go
		modifyHostNetworkOptionForContainer(false, podSandboxID, createConfig.HostConfig)
	}

	return nil
}

func (ds *dockerService) determinePodIPBySandboxID(sandboxID string) string {
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

		// Versions and feature support
		// ============================
		// Windows version == Windows Server, Version 1709,, Supports both sandbox and non-sandbox case
		// Windows version == Windows Server 2016   Support only non-sandbox case
		// Windows version < Windows Server 2016 is Not Supported

		// Sandbox support in Windows mandates CNI Plugin.
		// Presense of CONTAINER_NETWORK flag is considered as non-Sandbox cases here

		// Todo: Add a kernel version check for more validation

		if networkMode := os.Getenv("CONTAINER_NETWORK"); networkMode == "" {
			// Do not return any IP, so that we would continue and get the IP of the Sandbox
			ds.getIP(sandboxID, r)
		} else {
			// On Windows, every container that is created in a Sandbox, needs to invoke CNI plugin again for adding the Network,
			// with the shared container name as NetNS info,
			// This is passed down to the platform to replicate some necessary information to the new container

			//
			// This place is chosen as a hack for now, since getContainerIP would end up calling CNI's addToNetwork
			// That is why addToNetwork is required to be idempotent

			// Instead of relying on this call, an explicit call to addToNetwork should be
			// done immediately after ContainerCreation, in case of Windows only. TBD Issue # to handle this

			if containerIP := getContainerIP(r); containerIP != "" {
				return containerIP
			}
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

func getContainerIP(container *dockertypes.ContainerJSON) string {
	if container.NetworkSettings != nil {
		for _, network := range container.NetworkSettings.Networks {
			if network.IPAddress != "" {
				return network.IPAddress
			}
		}
	}
	return ""
}
