/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
)

// NamespaceGeneratorV1 supports stable generation of a namespace
type NamespaceGeneratorV1 struct {
	// Name of namespace
	Name string
}

// Ensure it supports the generator pattern that uses parameter injection
var _ Generator = &NamespaceGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &NamespaceGeneratorV1{}

// Generate returns a namespace using the specified parameters
func (g NamespaceGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(g.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	delegate := &NamespaceGeneratorV1{Name: params["name"]}
	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern
func (g NamespaceGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
	}
}

// StructuredGenerate outputs a namespace object using the configured fields
func (g *NamespaceGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := g.validate(); err != nil {
		return nil, err
	}
	namespace := &api.Namespace{}
	namespace.Name = g.Name
	return namespace, nil
}

// validate validates required fields are set to support structured generation
func (g *NamespaceGeneratorV1) validate() error {
	if len(g.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	return nil
}
