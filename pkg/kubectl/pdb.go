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

	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
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
		{"min-available", false},
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
		return nil, fmt.Errorf("expected string, found %T for 'name'", params["name"])
	}
	minAvailable, isString := params["min-available"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'min-available'", params["min-available"])
	}
	selector, isString := params["selector"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'selector'", params["selector"])
	}
	delegate := &PodDisruptionBudgetV1Generator{Name: name, MinAvailable: minAvailable, Selector: selector}
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a pod disruption budget object using the configured fields.
func (s *PodDisruptionBudgetV1Generator) StructuredGenerate() (runtime.Object, error) {
	if len(s.MinAvailable) == 0 {
		// defaulting behavior seen in Kubernetes 1.6 and below.
		s.MinAvailable = "1"
	}

	if err := s.validate(); err != nil {
		return nil, err
	}

	selector, err := metav1.ParseToLabelSelector(s.Selector)
	if err != nil {
		return nil, err
	}

	minAvailable := intstr.Parse(s.MinAvailable)
	return &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name: s.Name,
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
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
		return fmt.Errorf("the minimum number of available pods required must be specified")
	}
	return nil
}

// PodDisruptionBudgetV2Generator supports stable generation of a pod disruption budget.
type PodDisruptionBudgetV2Generator struct {
	Name           string
	MinAvailable   string
	MaxUnavailable string
	Selector       string
}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &PodDisruptionBudgetV2Generator{}

func (PodDisruptionBudgetV2Generator) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"min-available", false},
		{"max-unavailable", false},
		{"selector", true},
	}
}

func (s PodDisruptionBudgetV2Generator) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}

	name, isString := params["name"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'name'", params["name"])
	}

	minAvailable, isString := params["min-available"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'min-available'", params["min-available"])
	}

	maxUnavailable, isString := params["max-unavailable"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'max-unavailable'", params["max-unavailable"])
	}

	selector, isString := params["selector"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found %T for 'selector'", params["selector"])
	}
	delegate := &PodDisruptionBudgetV2Generator{Name: name, MinAvailable: minAvailable, MaxUnavailable: maxUnavailable, Selector: selector}
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a pod disruption budget object using the configured fields.
func (s *PodDisruptionBudgetV2Generator) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	selector, err := metav1.ParseToLabelSelector(s.Selector)
	if err != nil {
		return nil, err
	}

	if len(s.MaxUnavailable) > 0 {
		maxUnavailable := intstr.Parse(s.MaxUnavailable)
		return &policy.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name: s.Name,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				MaxUnavailable: &maxUnavailable,
				Selector:       selector,
			},
		}, nil
	}

	if len(s.MinAvailable) > 0 {
		minAvailable := intstr.Parse(s.MinAvailable)
		return &policy.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name: s.Name,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				MinAvailable: &minAvailable,
				Selector:     selector,
			},
		}, nil
	}

	return nil, err
}

// validate validates required fields are set to support structured generation.
func (s *PodDisruptionBudgetV2Generator) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.Selector) == 0 {
		return fmt.Errorf("a selector must be specified")
	}
	if len(s.MaxUnavailable) == 0 && len(s.MinAvailable) == 0 {
		return fmt.Errorf("one of min-available or max-unavailable must be specified")
	}
	if len(s.MaxUnavailable) > 0 && len(s.MinAvailable) > 0 {
		return fmt.Errorf("min-available and max-unavailable cannot be both specified")
	}
	return nil
}
