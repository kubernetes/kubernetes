/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package securitycontext

import (
	"k8s.io/kubernetes/pkg/api"

	docker "github.com/fsouza/go-dockerclient"
)

type SecurityContextProvider interface {
	// ModifyContainerConfig is called before the Docker createContainer call.
	// The security context provider can make changes to the Config with which
	// the container is created.
	ModifyContainerConfig(pod *api.Pod, container *api.Container, config *docker.Config)

	// ModifyHostConfig is called before the Docker runContainer call.
	// The security context provider can make changes to the HostConfig, affecting
	// security options, whether the container is privileged, volume binds, etc.
	// An error is returned if it's not possible to secure the container as requested
	// with a security context.
	ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *docker.HostConfig)
}

const (
	dockerLabelUser    string = "label:user"
	dockerLabelRole    string = "label:role"
	dockerLabelType    string = "label:type"
	dockerLabelLevel   string = "label:level"
	dockerLabelDisable string = "label:disable"
)
