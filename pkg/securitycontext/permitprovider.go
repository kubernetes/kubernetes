/*
Copyright 2014 Google Inc. All rights reserved.

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
	docker "github.com/fsouza/go-dockerclient"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
)

type permitProvider struct {
	*api.SecurityConstraints
}

func NewPermitSecurityContextProvider() SecurityContextProvider {
	return &permitProvider{
		SecurityConstraints: &api.SecurityConstraints{
			EnforcementPolicy: api.SecurityConstraintPolicyDisable,
		},
	}
}

func (p *permitProvider) ApplySecurityContext(pod *api.Pod)          {}
func (p *permitProvider) ValidateSecurityContext(pod *api.Pod) fielderrors.ValidationErrorList { return nil }
func (p *permitProvider) ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *docker.HostConfig) {
}
func (p *permitProvider) ModifyContainerConfig(pod *api.Pod, container *api.Container, config *docker.Config) error {
	return nil
}

