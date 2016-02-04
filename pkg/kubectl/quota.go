/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"strings"
)

// ResourceQuotaGeneratorV1 supports stable generation of a namespace
type ResourceQuotaGeneratorV1 struct {
	Name string
	Hard string
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern
func (g ResourceQuotaGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"hard", true},
	}
}

// Ensure it supports the generator pattern that uses parameter injection
var _ Generator = &ResourceQuotaGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &ResourceQuotaGeneratorV1{}

func (g ResourceQuotaGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
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

	delegate := &ResourceQuotaGeneratorV1{}
	delegate.Name = params["name"]
	delegate.Hard = params["hard"]
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a ResourceQuota object using the configured fields
func (g *ResourceQuotaGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := g.validate(); err != nil {
		return nil, err
	}

	resourceQuotaSpec, err := generateResourceQuotaSpecList(g.Hard)
	if err != nil {
		return nil, err
	}

	resourceQuota := &api.ResourceQuota{}
	resourceQuota.Name = g.Name
	resourceQuota.Spec.Hard = resourceQuotaSpec
	return resourceQuota, nil
}

func generateResourceQuotaSpecList(hard string) (resourceList api.ResourceList, err error) {

	defer func() {
		if p := recover(); p != nil {
			resourceList = nil
			err = fmt.Errorf("Invalid input %v", p)
		}
	}()

	resourceList = make(api.ResourceList)
	for _, keyValue := range strings.Split(hard, ",") {
		items := strings.Split(keyValue, "=")
		if len(items) != 2 {
			return nil, fmt.Errorf("invalid input %v, expected key=value", keyValue)
		}

		resourceList[api.ResourceName(items[0])] = resource.MustParse(items[1])
	}
	return
}

// validate validates required fields are set to support structured generation
func (r *ResourceQuotaGeneratorV1) validate() error {
	if len(r.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	return nil
}
