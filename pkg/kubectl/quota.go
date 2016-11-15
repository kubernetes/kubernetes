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
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

// ResourceQuotaGeneratorV1 supports stable generation of a resource quota
type ResourceQuotaGeneratorV1 struct {
	// The name of a quota object.
	Name string

	// The hard resource limit string before parsing.
	Hard string

	// The scopes of a quota object before parsing.
	Scopes string
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern
func (g ResourceQuotaGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"hard", true},
		{"scopes", false},
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
	delegate.Scopes = params["scopes"]
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a ResourceQuota object using the configured fields
func (g *ResourceQuotaGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := g.validate(); err != nil {
		return nil, err
	}

	resourceList, err := populateResourceList(g.Hard)
	if err != nil {
		return nil, err
	}

	scopes, err := parseScopes(g.Scopes)
	if err != nil {
		return nil, err
	}

	resourceQuota := &api.ResourceQuota{}
	resourceQuota.Name = g.Name
	resourceQuota.Spec.Hard = resourceList
	resourceQuota.Spec.Scopes = scopes
	return resourceQuota, nil
}

// validate validates required fields are set to support structured generation
func (r *ResourceQuotaGeneratorV1) validate() error {
	if len(r.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	return nil
}

func parseScopes(spec string) ([]api.ResourceQuotaScope, error) {
	// empty input gets a nil response to preserve generator test expected behaviors
	if spec == "" {
		return nil, nil
	}

	scopes := strings.Split(spec, ",")
	result := make([]api.ResourceQuotaScope, 0, len(scopes))
	for _, scope := range scopes {
		// intentionally do not verify the scope against the valid scope list. This is done by the apiserver anyway.

		if scope == "" {
			return nil, fmt.Errorf("invalid resource quota scope \"\"")
		}

		result = append(result, api.ResourceQuotaScope(scope))
	}
	return result, nil
}
