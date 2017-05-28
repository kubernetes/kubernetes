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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1"
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
