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
	"k8s.io/kubernetes/pkg/api/v1"

	docker "github.com/fsouza/go-dockerclient"
)

// ValidSecurityContextWithContainerDefaults creates a valid security context provider based on
// empty container defaults.  Used for testing.
func ValidSecurityContextWithContainerDefaults() *api.SecurityContext {
	priv := false
	return &api.SecurityContext{
		Capabilities: &api.Capabilities{},
		Privileged:   &priv,
	}
}

// V1ValidSecurityContextWithContainerDefaults is a temporary duplicate which
// return v1 SecuritContext.
func V1ValidSecurityContextWithContainerDefaults() *v1.SecurityContext {
	priv := false
	return &v1.SecurityContext{
		Capabilities: &v1.Capabilities{},
		Privileged:   &priv,
	}
}

// NewFakeSecurityContextProvider creates a new, no-op security context provider.
func NewFakeSecurityContextProvider() SecurityContextProvider {
	return FakeSecurityContextProvider{}
}

type FakeSecurityContextProvider struct{}

func (p FakeSecurityContextProvider) ModifyContainerConfig(pod *api.Pod, container *api.Container, config *docker.Config) {
}
func (p FakeSecurityContextProvider) ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *docker.HostConfig) {
}
