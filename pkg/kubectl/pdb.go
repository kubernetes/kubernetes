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

package kubectl

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// PodDisruptionBudgetV1Generator supports stable generation of a pod disruption budget.
type PodDisruptionBudgetV1Generator struct {
	Name         string
	MinAvailable string
	Selector     string
}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &PodDisruptionBudgetV1Generator{}

func (PodDisruptionBudgetV1Generator) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"mim-available", true},
		{"selector", true},
	}
}

func (s PodDisruptionBudgetV1Generator) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}
	name, isString := params["name"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, saw %v for 'name'", name)
	}
	minAvailable, isString := params["mim-available"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %v", minAvailable)
	}
	selector, isString := params["selecor"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %v", selector)
	}
	delegate := &PodDisruptionBudgetV1Generator{Name: name, MinAvailable: minAvailable, Selector: selector}
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a pod disruption budget object using the configured fields.
func (s *PodDisruptionBudgetV1Generator) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	selector, err := metav1.ParseToLabelSelector(s.Selector)
	if err != nil {
		return nil, err
	}

	return &policy.PodDisruptionBudget{
		ObjectMeta: api.ObjectMeta{
			Name: s.Name,
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: intstr.Parse(s.MinAvailable),
			Selector:     selector,
		},
	}, nil
}

// validate validates required fields are set to support structured generation.
func (s *PodDisruptionBudgetV1Generator) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.Selector) == 0 {
		return fmt.Errorf("a selector must be specified")
	}
	if len(s.MinAvailable) == 0 {
		return fmt.Errorf("the minimim number of available pods required must be specified")
	}
	return nil
}
