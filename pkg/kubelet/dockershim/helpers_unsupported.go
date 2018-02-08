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

package dockershim

import (
	"fmt"

	"github.com/blang/semver"
	dockertypes "github.com/docker/docker/api/types"
	"github.com/golang/glog"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

func DefaultMemorySwap() int64 {
	return -1
}

func (ds *dockerService) getSecurityOpts(seccompProfile string, separator rune) ([]string, error) {
	glog.Warningf("getSecurityOpts is unsupported in this build")
	return nil, nil
}

func (ds *dockerService) updateCreateConfig(
	createConfig *dockertypes.ContainerCreateConfig,
	config *runtimeapi.ContainerConfig,
	sandboxConfig *runtimeapi.PodSandboxConfig,
	podSandboxID string, securityOptSep rune, apiVersion *semver.Version) error {
	glog.Warningf("updateCreateConfig is unsupported in this build")
	return nil
}

func (ds *dockerService) determinePodIPBySandboxID(uid string) string {
	glog.Warningf("determinePodIPBySandboxID is unsupported in this build")
	return ""
}

func getNetworkNamespace(c *dockertypes.ContainerJSON) (string, error) {
	return "", fmt.Errorf("unsupported platform")
}

// applyExperimentalCreateConfig applys experimental configures from sandbox annotations.
func applyExperimentalCreateConfig(createConfig *dockertypes.ContainerCreateConfig, annotations map[string]string) {
}
