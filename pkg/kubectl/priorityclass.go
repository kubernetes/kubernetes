/*
Copyright 2017 The Kubernetes Authors.

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

	scheduling "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// PriorityClassV1Generator supports stable generation of a priorityClass.
type PriorityClassV1Generator struct {
	Name          string
	Value         int32
	GlobalDefault bool
	Description   string
}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &PriorityClassV1Generator{}

func (PriorityClassV1Generator) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"value", true},
		{"global-default", false},
		{"description", false},
	}
}

func (s PriorityClassV1Generator) Generate(params map[string]interface{}) (runtime.Object, error) {
	if err := ValidateParams(s.ParamNames(), params); err != nil {
		return nil, err
	}

	name, found := params["name"].(string)
	if !found {
		return nil, fmt.Errorf("expected string, saw %v for 'name'", name)
	}

	value, found := params["value"].(int32)
	if !found {
		return nil, fmt.Errorf("expected int32, found %v", value)
	}

	globalDefault, found := params["global-default"].(bool)
	if !found {
		return nil, fmt.Errorf("expected bool, found %v", globalDefault)
	}

	description, found := params["description"].(string)
	if !found {
		return nil, fmt.Errorf("expected string, found %v", description)
	}
	delegate := &PriorityClassV1Generator{Name: name, Value: value, GlobalDefault: globalDefault, Description: description}
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a priorityClass object using the configured fields.
func (s *PriorityClassV1Generator) StructuredGenerate() (runtime.Object, error) {
	return &scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: s.Name,
		},
		Value:         s.Value,
		GlobalDefault: s.GlobalDefault,
		Description:   s.Description,
	}, nil
}
