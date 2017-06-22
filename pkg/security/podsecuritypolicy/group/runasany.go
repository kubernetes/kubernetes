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

package group

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
)

// runAsAny implements the GroupStrategy interface.
type runAsAny struct {
}

var _ GroupStrategy = &runAsAny{}

// NewRunAsAny provides a new RunAsAny strategy.
func NewRunAsAny() (GroupStrategy, error) {
	return &runAsAny{}, nil
}

// Generate creates the group based on policy rules.  This strategy returns an empty slice.
func (s *runAsAny) Generate(pod *api.Pod) ([]int64, error) {
	return []int64{}, nil
}

// Generate a single value to be applied.  This is used for FSGroup.  This strategy returns nil.
func (s *runAsAny) GenerateSingle(pod *api.Pod) (*int64, error) {
	return nil, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *runAsAny) Validate(pod *api.Pod, groups []int64) field.ErrorList {
	return field.ErrorList{}

}
