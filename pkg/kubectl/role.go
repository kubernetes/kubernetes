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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// RoleGeneratorV1 supports stable generation of a role.
type RoleGeneratorV1 struct {
	// Name of role (required)
	Name string
	// Verbs is a list of Verbs that apply to ALL the ResourceKinds contained in this rule (required)
	Verbs []string
	// Resources is a list of resources this rule applies to (required)
	Resources []string
	// APIGroups is the name of the APIGroup that contains the resources (optional)
	APIGroups []string
	// ResourceNames is an optional white list of names that the rule applies to (optional)
	ResourceNames []string
	// NonResourceURLs is a set of partial urls that a user should have access to (optional)
	NonResourceURLs []string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &RoleGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &RoleGeneratorV1{}

// Generate returns a role using the specified parameters.
func (s RoleGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}

	delegate := &RoleGeneratorV1{}

	name, isString := genericParams["name"].(string)
	if !isString {
		return nil, fmt.Errorf("expected string, found: %v", name)
	}
	delegate.Name = name

	verbs, isArray := genericParams["verb"].([]string)
	if !isArray {
		return nil, fmt.Errorf("expected []string, found: %v", verbs)
	}
	delegate.Verbs = verbs

	resources, isArray := genericParams["resource"].([]string)
	if !isArray {
		return nil, fmt.Errorf("expected []string, found: %v", resources)
	}
	delegate.Resources = resources

	// APIGroup is required, and "" indicates the core API Group.
	apiGroupStrings, found := genericParams["api-group"]
	if found {
		apiGroups, isArray := apiGroupStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found: %v", apiGroupStrings)
		}
		delegate.APIGroups = apiGroups
	} else {
		delegate.APIGroups = []string{""}
	}

	resourceNameStrings, found := genericParams["resource-name"]
	if found {
		resourceNames, isArray := resourceNameStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", resourceNameStrings)
		}
		delegate.ResourceNames = resourceNames
	}

	nonResourceURLStrings, found := genericParams["non-resource-url"]
	if found {
		nonResourceURLs, isArray := nonResourceURLStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", nonResourceURLStrings)
		}
		delegate.NonResourceURLs = nonResourceURLs
	}

	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s RoleGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"verb", true},
		{"resource", true},
		{"api-group", false},
		{"resource-name", false},
		{"non-resource-url", false},
	}
}

// StructuredGenerate outputs a role object using the configured fields.
func (s RoleGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	role := &rbac.Role{}
	rule := rbac.PolicyRule{}

	role.Name = s.Name
	rule.Verbs = s.Verbs
	rule.Resources = s.Resources

	// APIGroup is required, and "" indicates the core API Group.
	if len(s.APIGroups) == 0 {
		rule.APIGroups = []string{""}
	} else {
		rule.APIGroups = s.APIGroups
	}

	rule.ResourceNames = s.ResourceNames
	rule.NonResourceURLs = s.NonResourceURLs

	// At present, we only allow creating one single rule by using 'kubectl create role' command.
	role.Rules = []rbac.PolicyRule{rule}

	return role, nil
}

// validate validates required fields are set to support structured generation.
func (s RoleGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}
	if len(s.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}
	return nil
}
