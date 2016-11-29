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

package dockertools

import (
	"os"

	"k8s.io/kubernetes/pkg/api/v1"

	dockertypes "github.com/docker/engine-api/types"
)

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

func getNetworkingMode() string {
	// Allow override via env variable. Otherwise, use a default "kubenet" network
	netMode := os.Getenv("CONTAINER_NETWORK")
	if netMode == "" {
		netMode = "kubenet"
	}
	return netMode
}

// Infrastructure containers are not supported on Windows. For this reason, we
// make sure to not grab the infra container's IP for the pod.
func containerProvidesPodIP(name *KubeletContainerName) bool {
	return name.ContainerName != PodInfraContainerName
}

// Returns nil as both Seccomp and AppArmor security options are not valid on Windows
func (dm *DockerManager) getSecurityOpts(pod *v1.Pod, ctrName string) ([]dockerOpt, error) {
	return nil, nil
}
