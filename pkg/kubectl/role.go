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

	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

var (
	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "deletecollection", "use", "bind", "impersonate"}

	// Specialized verbs and GroupResources
	specialVerbs = map[string][]schema.GroupResource{
		"use": {
			{
				Group:    "extensions",
				Resource: "podsecuritypolicies",
			},
		},
		"bind": {
			{
				Group:    "rbac.authorization.k8s.io",
				Resource: "roles",
			},
			{
				Group:    "rbac.authorization.k8s.io",
				Resource: "clusterroles",
			},
		},
		"impersonate": {
			{
				Group:    "",
				Resource: "users",
			},
			{
				Group:    "",
				Resource: "serviceaccounts",
			},
			{
				Group:    "",
				Resource: "groups",
			},
			{
				Group:    "authentication.k8s.io",
				Resource: "userextras",
			},
		},
	}
)

// RoleGeneratorV1 supports stable generation of a role.
type RoleGeneratorV1 struct {
	Mapper meta.RESTMapper
	// Name of roleBinding (required)
	Name string
	// Verbs for the role
	Verbs []string
	// Resources for the role
	Resources []string
	// ResourceNames for the roleBinding
	ResourceNames []string
}

type ResourceOptions struct {
	Group       string
	Resource    string
	SubResource string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &RoleGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &RoleGeneratorV1{}

// Generate returns a roleBinding using the specified parameters.
func (s RoleGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &RoleGeneratorV1{}
	verbStrings, found := genericParams["verb"]
	if found {
		fromFileArray, isArray := verbStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", verbStrings)
		}
		delegate.Verbs = fromFileArray
		delete(genericParams, "verb")
	}
	resourceStrings, found := genericParams["resource"]
	if found {
		fromLiteralArray, isArray := resourceStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", resourceStrings)
		}
		delegate.Resources = fromLiteralArray
		delete(genericParams, "resource")
	}
	rsNameStrings, found := genericParams["resource-name"]
	if found {
		fromLiteralArray, isArray := rsNameStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", rsNameStrings)
		}
		delegate.ResourceNames = fromLiteralArray
		delete(genericParams, "resource-name")
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

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s RoleGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"verb", false},
		{"resource", false},
		{"resource-name", false},
	}
}

// StructuredGenerate outputs a role object using the configured fields.
func (s RoleGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	role := &rbac.Role{}
	role.Name = s.Name

	// Remove duplicate verbs.
	verbs := []string{}
	for _, v := range s.Verbs {
		// VerbAll respresents all kinds of verbs.
		if v == "*" {
			verbs = []string{"*"}
			break
		}
		if !arrayContains(verbs, v) {
			verbs = append(verbs, v)
		}
	}
	s.Verbs = verbs

	// Support resource.group pattern. If no API Group specified, use "" as core API Group.
	// e.g. --resource=pods,deployments.extensions
	ro := []ResourceOptions{}
	for _, r := range s.Resources {
		sections := strings.SplitN(r, "/", 2)

		resource := ResourceOptions{}
		if len(sections) == 2 {
			resource.SubResource = sections[1]
		}

		parts := strings.SplitN(sections[0], ".", 2)
		if len(parts) == 2 {
			resource.Group = parts[1]
		}
		resource.Resource = parts[0]

		ro = append(ro, resource)
	}

	if err := s.validateResource(ro); err != nil {
		return nil, err
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range s.ResourceNames {
		if !arrayContains(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	s.ResourceNames = resourceNames

	rules, err := GenerateResourcePolicyRules(s.Mapper, s.Verbs, ro, s.ResourceNames, []string{})
	if err != nil {
		return nil, err
	}

	role.Rules = rules

	return role, nil
}

func (s *RoleGeneratorV1) validate() error {
	if s.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	if len(s.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	for _, v := range s.Verbs {
		if !arrayContains(validResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
	}

	// validate resources.
	if len(s.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}

	return nil
}

func (s RoleGeneratorV1) validateResource(ro []ResourceOptions) error {
	for _, r := range ro {
		if len(r.Resource) == 0 {
			return fmt.Errorf("resource must be specified if apiGroup/subresource specified")
		}

		resource := schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}
		groupVersionResource, err := s.Mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err == nil {
			resource = groupVersionResource
		}

		for _, v := range s.Verbs {
			if groupResources, ok := specialVerbs[v]; ok {
				match := false
				for _, extra := range groupResources {
					if resource.Resource == extra.Resource && resource.Group == extra.Group {
						match = true
						err = nil
						break
					}
				}
				if !match {
					return fmt.Errorf("can not perform '%s' on '%s' in group '%s'", v, resource.Resource, resource.Group)
				}
			}
		}

		if err != nil {
			return err
		}
	}
	return nil
}

func GenerateResourcePolicyRules(mapper meta.RESTMapper, verbs []string, resources []ResourceOptions, resourceNames []string, nonResourceURLs []string) ([]rbac.PolicyRule, error) {
	// groupResourceMapping is a apigroup-resource map. The key of this map is api group, while the value
	// is a string array of resources under this api group.
	// E.g.  groupResourceMapping = {"extensions": ["replicasets", "deployments"], "batch":["jobs"]}
	groupResourceMapping := map[string][]string{}

	// This loop does the following work:
	// 1. Constructs groupResourceMapping based on input resources.
	// 2. Prevents pointing to non-existent resources.
	// 3. Transfers resource short name to long name. E.g. rs.extensions is transferred to replicasets.extensions
	for _, r := range resources {
		resource := schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}
		groupVersionResource, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err == nil {
			resource = groupVersionResource
		}

		if len(r.SubResource) > 0 {
			resource.Resource = resource.Resource + "/" + r.SubResource
		}
		if !arrayContains(groupResourceMapping[resource.Group], resource.Resource) {
			groupResourceMapping[resource.Group] = append(groupResourceMapping[resource.Group], resource.Resource)
		}
	}

	// Create separate rule for each of the api group.
	rules := []rbac.PolicyRule{}
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbac.PolicyRule{}
		rule.Verbs = verbs
		rule.Resources = groupResourceMapping[g]
		rule.APIGroups = []string{g}
		rule.ResourceNames = resourceNames
		rules = append(rules, rule)
	}

	if len(nonResourceURLs) > 0 {
		rule := rbac.PolicyRule{}
		rule.Verbs = verbs
		rule.NonResourceURLs = nonResourceURLs
		rules = append(rules, rule)
	}

	return rules, nil
}
