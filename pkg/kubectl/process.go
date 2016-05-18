/*
Copyright 2016 The Kubernetes Authors All rights reserved.
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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/runtime"
)

// Processor provides an interface for resources that can be processed.
type Processor interface {
	Process(namespace string, obj runtime.Object) (*api.List, error)
}

func ProcessorFor(kind unversioned.GroupKind, c clientset.Interface) (Processor, error) {
	switch kind {
	case extensions.Kind("TemplateParameters"):
		return &TemplateProcessor{c}, nil
	}
	return nil, fmt.Errorf("no processor has been implemented for %q", kind)
}

type TemplateProcessor struct {
	clientset.Interface
}

func (r *TemplateProcessor) Process(namespace string, obj runtime.Object) (*api.List, error) {
	tp := obj.(*extensions.TemplateParameters)
	return r.Extensions().Templates(namespace).Process(tp)
}

// TODO: Generalize generalize the generator code since it is copied (with changes) from configmap and secrets
// TemplateParametersV1Beta1 supports stable generation of TemplateParameters
type TemplateParametersGeneratorV1Beta1 struct {
	// Name of Template (required)
	Name string
	// LiteralSources to derive the TemplateParameters from (optional)
	LiteralSources []string
}

// Ensure it supports the generator pattern that uses parameter injection
var _ Generator = &TemplateParametersGeneratorV1Beta1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &TemplateParametersGeneratorV1Beta1{}

// Override=kubectl.Generator
func (s TemplateParametersGeneratorV1Beta1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	delegate := TemplateParametersGeneratorV1Beta1{}
	fromLiteralStrings, found := genericParams["from-literal"]
	if found {
		fromLiteralArray, isArray := fromLiteralStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromLiteralArray)
		}
		delegate.LiteralSources = fromLiteralArray
		delete(genericParams, "from-literal")
	}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	delegate.Name = params["name"]
	return delegate.StructuredGenerate()
}

// Override=kubectl.StructuredGenerator
func (s TemplateParametersGeneratorV1Beta1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	templateParameters := &extensions.TemplateParameters{}
	templateParameters.Name = s.Name
	if len(s.LiteralSources) > 0 {
		paramsMap, err := literalSourcesToMap(s.LiteralSources)
		if err != nil {
			return nil, err
		}
		templateParameters.ParameterValues = paramsMap
	}
	return templateParameters, nil
}

// Override=kubectl.Generator
func (s TemplateParametersGeneratorV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"from-literal", false},
	}
}

// validate validates required fields are set to support structured generation.
func (s TemplateParametersGeneratorV1Beta1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	// Actual Parameter validation done in the Server
	return nil
}

// TODO: De-Dup this with configmap parsing
// handleTemplateParametersFromLiteralSources adds the specified literal source
// information into the provided templateParameters.
func literalSourcesToMap(literalSources []string) (map[string]string, error) {
	result := map[string]string{}
	for _, literalSource := range literalSources {
		keyName, value, err := parseLiteralSource(literalSource)
		if err != nil {
			return nil, err
		}
		result[keyName] = value
	}
	return result, nil
}
