/*
Copyright 2014 The Kubernetes Authors.

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

	dockercontainer "github.com/docker/engine-api/types/container"
)

type SecurityContextProvider interface {
	// ModifyContainerConfig is called before the Docker createContainer call.
	// The security context provider can make changes to the Config with which
	// the container is created.
	ModifyContainerConfig(pod *api.Pod, container *api.Container, config *dockercontainer.Config)

	// ModifyHostConfig is called before the Docker createContainer call.
	// The security context provider can make changes to the HostConfig, affecting
	// security options, whether the container is privileged, volume binds, etc.
	// An error is returned if it's not possible to secure the container as requested
	// with a security context.
	//
	// - pod: the pod to modify the docker hostconfig for
	// - container: the container to modify the hostconfig for
	// - supplementalGids: additional supplemental GIDs associated with the pod's volumes
	ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *dockercontainer.HostConfig, supplementalGids []int64)
}

const (
	DockerLabelUser    string = "label:user"
	DockerLabelRole    string = "label:role"
	DockerLabelType    string = "label:type"
	DockerLabelLevel   string = "label:level"
	DockerLabelDisable string = "label:disable"
)
