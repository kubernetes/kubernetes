// +build !linux,!windows

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
}

func DefaultMemorySwap() int64 {
	return -1
}

func getContainerIP(container *dockertypes.ContainerJSON) string {
	return ""
}

func getNetworkingMode() string {
	return ""
}

func containerProvidesPodIP(name *KubeletContainerName) bool {
	return false
}

// Returns nil as both Seccomp and AppArmor security options are not valid on Windows
func (dm *DockerManager) getSecurityOpts(pod *v1.Pod, ctrName string) ([]dockerOpt, error) {
	return nil, nil
}
