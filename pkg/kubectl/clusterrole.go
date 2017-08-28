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
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// Valid nonResource verb list for validation.
var validNonResourceVerbs = []string{"*", "get", "post", "put", "delete", "patch", "head", "options"}

// ClusterRoleGeneratorV1 supports stable generation of a role.
type ClusterRoleGeneratorV1 struct {
	Mapper meta.RESTMapper
	// Name of clusterrole (required)
	Name string
	// Verbs for the clusterrole
	Verbs []string
	// Resources for the clusterrole
	Resources []string
	// ResourceNames for the clusterrole
	ResourceNames []string
	// NonResourceURLs for the clusterrole
	NonResourceURLs []string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &ClusterRoleGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &ClusterRoleGeneratorV1{}

// Generate returns a roleBinding using the specified parameters.
func (s ClusterRoleGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &ClusterRoleGeneratorV1{}
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
	nrUrlStrings, found := genericParams["non-resource-url"]
	if found {
		fromLiteralArray, isArray := nrUrlStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", nrUrlStrings)
		}
		delegate.ResourceNames = fromLiteralArray
		delete(genericParams, "non-resource-url")
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
func (s ClusterRoleGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"verb", false},
		{"resource", false},
		{"resource-name", false},
		{"non-resource-url", false},
	}
}

// StructuredGenerate outputs a role object using the configured fields.
func (s ClusterRoleGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	clusterRole := &rbac.ClusterRole{}
	clusterRole.Name = s.Name

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

	// Remove duplicate NonResourceURLs
	NonResourceURLs := []string{}
	for _, n := range s.NonResourceURLs {
		if !arrayContains(NonResourceURLs, n) {
			NonResourceURLs = append(NonResourceURLs, n)
		}
	}
	s.NonResourceURLs = NonResourceURLs

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
	if len(ro) != 0 {
		if err := s.validateResource(ro); err != nil {
			return nil, err
		}
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range s.ResourceNames {
		if !arrayContains(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	s.ResourceNames = resourceNames

	rules, err := GenerateResourcePolicyRules(s.Mapper, s.Verbs, ro, s.ResourceNames, s.NonResourceURLs)
	if err != nil {
		return nil, err
	}

	clusterRole.Rules = rules

	return clusterRole, nil
}

func (s ClusterRoleGeneratorV1) validate() error {
	if s.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	if len(s.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	if len(s.Resources) == 0 && len(s.NonResourceURLs) == 0 {
		return fmt.Errorf("one of resource or nonResourceURL must be specified")
	}

	// validate resources
	if len(s.Resources) > 0 {
		for _, v := range s.Verbs {
			if !arrayContains(validResourceVerbs, v) {
				return fmt.Errorf("invalid verb: '%s'", v)
			}
		}
	}

	//validate non-resource-url
	if len(s.NonResourceURLs) > 0 {
		for _, v := range s.Verbs {
			if !arrayContains(validNonResourceVerbs, v) {
				return fmt.Errorf("invalid verb: '%s' for nonResourceURL", v)
			}
		}
		for _, nonResourceURL := range s.NonResourceURLs {
			if nonResourceURL == "*" {
				continue
			}

			if nonResourceURL == "" || !strings.HasPrefix(nonResourceURL, "/") {
				return fmt.Errorf("nonResourceURL should start with /")
			}

			if strings.ContainsRune(nonResourceURL[:len(nonResourceURL)-1], '*') {
				return fmt.Errorf("nonResourceURL only supports wildcard matches when '*' is at the end")
			}
		}
	}

	return nil
}

func (s ClusterRoleGeneratorV1) validateResource(ro []ResourceOptions) error {
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
