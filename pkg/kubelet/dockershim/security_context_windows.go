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
	dockercontainer "github.com/docker/docker/api/types/container"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	knetwork "k8s.io/kubernetes/pkg/kubelet/dockershim/network"
)

// applySandboxSecurityContext updates docker sandbox options according to security context.
func applySandboxSecurityContext(wc *runtimeapi.WindowsPodSandboxConfig, config *dockercontainer.Config, hc *dockercontainer.HostConfig, network *knetwork.PluginManager, separator rune) error {
	if wc == nil {
		return nil
	}

	var sc *runtimeapi.WindowsContainerSecurityContext
	if wc.SecurityContext != nil {
		sc = &runtimeapi.WindowsContainerSecurityContext{
			RunAsUsername: wc.SecurityContext.RunAsUser,
		}
	}

	err := modifyContainerConfig(sc, config)
	if err != nil {
		return err
	}
	return nil
}

// modifyContainerConfig applies container security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeapi.WindowsContainerSecurityContext, config *dockercontainer.Config) error {
	if sc == nil {
		return nil
	}
	if sc.RunAsUsername != "" {
		config.User = sc.RunAsUsername
	}
	return nil
}
