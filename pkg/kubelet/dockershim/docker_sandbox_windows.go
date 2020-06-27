// +build windows,!dockerless

/*
Copyright 2016 The Kubernetes Authors.

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
	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

// applySandboxOptions applies WindowsPodSandboxConfig to dockercontainer.HostConfig and dockercontainer.ContainerCreateConfig.
func (ds *dockerService) applySandboxOptions(hc *dockercontainer.HostConfig, c *runtimeapi.PodSandboxConfig, createConfig *dockertypes.ContainerCreateConfig, image string, separator rune) error {
	wc := c.GetWindows()
	if wc == nil {
		return nil
	}
	// Apply security context.
	if err := applySandboxSecurityContext(wc, createConfig.Config, hc, ds.network, separator); err != nil {
		return err
	}

	return nil
}
