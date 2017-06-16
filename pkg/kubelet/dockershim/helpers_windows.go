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
	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
)

func DefaultMemorySwap() int64 {
	return 0
}

func (ds *dockerService) getSecurityOpts(containerName string, sandboxConfig *runtimeapi.PodSandboxConfig, separator rune) ([]string, error) {
	hasSeccompSetting := false
	annotations := sandboxConfig.GetAnnotations()
	if _, ok := annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName]; !ok {
		_, hasSeccompSetting = annotations[v1.SeccompPodAnnotationKey]
	} else {
		hasSeccompSetting = true
	}

	if hasSeccompSetting {
		glog.Warningf("seccomp annotations found, but it is not supported on windows")
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
	}

	return nil
}

func (ds *dockerService) determinePodIPBySandboxID(sandboxID string) string {
	opts := dockertypes.ContainerListOptions{
		All:    true,
		Filter: dockerfilters.NewArgs(),
	}

	f := newDockerFilter(&opts.Filter)
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
		if containerIP := getContainerIP(r); containerIP != "" {
			return containerIP
		}
	}

	return ""
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
