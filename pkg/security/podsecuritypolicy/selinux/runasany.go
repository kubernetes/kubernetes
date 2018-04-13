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

package selinux

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
)

// runAsAny implements the SELinuxStrategy interface.
type runAsAny struct{}

var _ SELinuxStrategy = &runAsAny{}

// NewRunAsAny provides a strategy that will return the configured se linux context or nil.
func NewRunAsAny(options *policy.SELinuxStrategyOptions) (SELinuxStrategy, error) {
	return &runAsAny{}, nil
}

// Generate creates the SELinuxOptions based on constraint rules.
func (s *runAsAny) Generate(pod *api.Pod, container *api.Container) (*api.SELinuxOptions, error) {
	return nil, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *runAsAny) Validate(fldPath *field.Path, _ *api.Pod, _ *api.Container, options *api.SELinuxOptions) field.ErrorList {
	return field.ErrorList{}
}
