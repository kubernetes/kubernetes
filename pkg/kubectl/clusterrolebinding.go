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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// ClusterRoleBindingGeneratorV1 supports stable generation of a clusterRoleBinding.
type ClusterRoleBindingGeneratorV1 struct {
	// Name of clusterRoleBinding (required)
	Name string
	// ClusterRole for the clusterRoleBinding (required)
	ClusterRole string
	// Users to derive the clusterRoleBinding from (optional)
	Users []string
	// Groups to derive the clusterRoleBinding from (optional)
	Groups []string
	// ServiceAccounts to derive the clusterRoleBinding from in namespace:name format(optional)
	ServiceAccounts []string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &ClusterRoleBindingGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &ClusterRoleBindingGeneratorV1{}

// Generate returns a clusterRoleBinding using the specified parameters.
func (s ClusterRoleBindingGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &ClusterRoleBindingGeneratorV1{}
	userStrings, found := genericParams["user"]
	if found {
		fromFileArray, isArray := userStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", userStrings)
		}
		delegate.Users = fromFileArray
		delete(genericParams, "user")
	}
	groupStrings, found := genericParams["group"]
	if found {
		fromLiteralArray, isArray := groupStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", groupStrings)
		}
		delegate.Groups = fromLiteralArray
		delete(genericParams, "group")
	}
	saStrings, found := genericParams["serviceaccount"]
	if found {
		fromLiteralArray, isArray := saStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", saStrings)
		}
		delegate.ServiceAccounts = fromLiteralArray
		delete(genericParams, "serviceaccount")
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
	delegate.ClusterRole = params["clusterrole"]
	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s ClusterRoleBindingGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"clusterrole", false},
		{"user", false},
		{"group", false},
		{"serviceaccount", false},
	}
}

// StructuredGenerate outputs a clusterRoleBinding object using the configured fields.
func (s ClusterRoleBindingGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	clusterRoleBinding := &rbac.ClusterRoleBinding{}
	clusterRoleBinding.Name = s.Name
	clusterRoleBinding.RoleRef = rbac.RoleRef{
		APIGroup: rbac.GroupName,
		Kind:     "ClusterRole",
		Name:     s.ClusterRole,
	}
	for _, user := range s.Users {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbac.Subject{
			Kind:     rbac.UserKind,
			APIGroup: rbac.GroupName,
			Name:     user,
		})
	}
	for _, group := range s.Groups {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbac.Subject{
			Kind:     rbac.GroupKind,
			APIGroup: rbac.GroupName,
			Name:     group,
		})
	}
	for _, sa := range s.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbac.Subject{
			Kind:      rbac.ServiceAccountKind,
			APIGroup:  "",
			Namespace: tokens[0],
			Name:      tokens[1],
		})
	}

	return clusterRoleBinding, nil
}

// validate validates required fields are set to support structured generation.
func (s ClusterRoleBindingGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.ClusterRole) == 0 {
		return fmt.Errorf("clusterrole must be specified")
	}
	return nil
}
