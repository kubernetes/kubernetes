// +build linux

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

package dockertools

import (
	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"

	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// These two functions are OS specific (for now at least)
func updateHostConfig(hc *dockercontainer.HostConfig, opts *kubecontainer.RunContainerOptions) {
	// no-op, there is a windows implementation that is different.
}

func DefaultMemorySwap() int64 {
	return 0
}

func getContainerIP(container *dockertypes.ContainerJSON) string {
	result := ""
	if container.NetworkSettings != nil {
		result = container.NetworkSettings.IPAddress

		// Fall back to IPv6 address if no IPv4 address is present
		if result == "" {
			result = container.NetworkSettings.GlobalIPv6Address
		}
	}
	return result
}

// We don't want to override the networking mode on Linux.
func getNetworkingMode() string { return "" }

// Returns true if the container name matches the infrastructure's container name
func containerProvidesPodIP(name *KubeletContainerName) bool {
	return name.ContainerName == PodInfraContainerName
}

// Returns Seccomp and AppArmor Security options
func (dm *DockerManager) getSecurityOpts(pod *v1.Pod, ctrName string) ([]dockerOpt, error) {
	var securityOpts []dockerOpt
	if seccompOpts, err := dm.getSeccompOpts(pod, ctrName); err != nil {
		return nil, err
	} else {
		securityOpts = append(securityOpts, seccompOpts...)
	}

	if appArmorOpts, err := dm.getAppArmorOpts(pod, ctrName); err != nil {
		return nil, err
	} else {
		securityOpts = append(securityOpts, appArmorOpts...)
	}

	return securityOpts, nil
}
