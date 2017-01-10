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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
	// APIGroups is the name of the APIGroup that contains the resources (required)
	APIGroups []string
	// ResourceNames is an optional white list of names that the rule applies to (optional)
	ResourceNames []string
	// Mapper is used to validate resources.
	Mapper meta.RESTMapper
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &RoleGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &RoleGeneratorV1{}

// Valid resource verb list for validation.
var validResourceVerbs []string = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "redirect", "deletecollection"}

// Valid non-resource verb list for validation.
var validNonResourceVerbs []string = []string{"get", "post", "put", "delete"}

// Generate returns a role using the specified parameters.
func (s *RoleGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}

	delegate := &RoleGeneratorV1{Mapper: s.Mapper}

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

	apiGroups, isArray := genericParams["api-group"].([]string)
	if !isArray {
		return nil, fmt.Errorf("expected []string, found: %v", apiGroups)
	}
	delegate.APIGroups = apiGroups

	resourceNameStrings, found := genericParams["resource-name"]
	if found {
		resourceNames, isArray := resourceNameStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", resourceNameStrings)
		}
		delegate.ResourceNames = resourceNames
	}

	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s *RoleGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"verb", true},
		{"resource", true},
		{"api-group", true},
		{"resource-name", false},
	}
}

// StructuredGenerate outputs a role object using the configured fields.
func (s *RoleGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	role := &rbac.Role{}
	rule := rbac.PolicyRule{}

	role.Name = s.Name
	rule.Verbs = s.Verbs
	rule.Resources = s.Resources
	rule.APIGroups = s.APIGroups
	rule.ResourceNames = s.ResourceNames

	// At present, we only allow creating one single rule by using 'kubectl create role' command.
	role.Rules = []rbac.PolicyRule{rule}

	return role, nil
}

// validate validates required fields are set to support structured generation.
func (s *RoleGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}

	err := s.validateVerbs()
	if err != nil {
		return err
	}

	err = s.validateResources()
	if err != nil {
		return err
	}

	if len(s.APIGroups) == 0 {
		return fmt.Errorf("at least one API group must be specified")
	}

	return s.validateResourceNames()
}

// validateVerbs validates verbs.
func (s *RoleGeneratorV1) validateVerbs() error {
	if len(s.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	containsVerbAll := false
	for _, v := range s.Verbs {
		if !arrayContains(validResourceVerbs, v) && !arrayContains(validNonResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
		if v == "*" {
			containsVerbAll = true
		}
	}

	// VerbAll respresents all kinds of verbs.
	if containsVerbAll {
		s.Verbs = []string{"*"}
	}

	return nil
}

// validateResources validates resources.
func (s *RoleGeneratorV1) validateResources() error {
	if len(s.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}

	if s.Mapper == nil {
		return fmt.Errorf("no REST Mapper found to validate resources")
	}

	for _, r := range s.Resources {
		resource, err := s.Mapper.ResourceFor(schema.GroupVersionResource{Resource: r})
		if err != nil {
			return err
		}
		if !arrayContains(s.APIGroups, resource.Group) {
			return fmt.Errorf("resource '%s' is under api group '%s', corresponding API group is not provided", r, resource.Group)
		}
	}

	return nil
}

// validateResourceNames validates resource names.
func (s *RoleGeneratorV1) validateResourceNames() error {
	if len(s.ResourceNames) > 0 && len(s.Resources) > 1 {
		return fmt.Errorf("resource name(s) can not be applied to multiple resources")
	}

	return nil
}
