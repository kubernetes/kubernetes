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

// RoleBindingGeneratorV1 supports stable generation of a roleBinding.
type RoleBindingGeneratorV1 struct {
	// Name of roleBinding (required)
	Name string
	// ClusterRole for the roleBinding
	ClusterRole string
	// Role for the roleBinding
	Role string
	// Users to derive the roleBinding from (optional)
	Users []string
	// Groups to derive the roleBinding from (optional)
	Groups []string
	// ServiceAccounts to derive the roleBinding from in namespace:name format(optional)
	ServiceAccounts []string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &RoleBindingGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &RoleBindingGeneratorV1{}

// Generate returns a roleBinding using the specified parameters.
func (s RoleBindingGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &RoleBindingGeneratorV1{}
	fromFileStrings, found := genericParams["user"]
	if found {
		fromFileArray, isArray := fromFileStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
		delegate.Users = fromFileArray
		delete(genericParams, "user")
	}
	fromLiteralStrings, found := genericParams["group"]
	if found {
		fromLiteralArray, isArray := fromLiteralStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
		delegate.Groups = fromLiteralArray
		delete(genericParams, "group")
	}
	fromSAStrings, found := genericParams["serviceaccount"]
	if found {
		fromLiteralArray, isArray := fromSAStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
		delegate.ServiceAccounts = fromLiteralArray
		delete(genericParams, "serviceaccounts")
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
	delegate.Role = params["role"]
	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s RoleBindingGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"clusterrole", false},
		{"role", false},
		{"user", false},
		{"group", false},
		{"serviceaccount", false},
		{"force", false},
	}
}

// StructuredGenerate outputs a roleBinding object using the configured fields.
func (s RoleBindingGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	roleBinding := &rbac.RoleBinding{}
	roleBinding.Name = s.Name

	switch {
	case len(s.Role) > 0:
		roleBinding.RoleRef = rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     s.Role,
		}
	case len(s.ClusterRole) > 0:
		roleBinding.RoleRef = rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     s.ClusterRole,
		}
	}

	for _, user := range s.Users {
		roleBinding.Subjects = append(roleBinding.Subjects, rbac.Subject{
			Kind:       rbac.UserKind,
			APIVersion: "rbac/v1alpha1",
			Name:       user,
		})
	}
	for _, group := range s.Groups {
		roleBinding.Subjects = append(roleBinding.Subjects, rbac.Subject{
			Kind:       rbac.GroupKind,
			APIVersion: "rbac/v1alpha1",
			Name:       group,
		})
	}
	for _, sa := range s.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		roleBinding.Subjects = append(roleBinding.Subjects, rbac.Subject{
			Kind:      rbac.ServiceAccountKind,
			Namespace: tokens[0],
			Name:      tokens[1],
		})
	}

	return roleBinding, nil
}

// validate validates required fields are set to support structured generation.
func (s RoleBindingGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if (len(s.ClusterRole) == 0) == (len(s.Role) == 0) {
		return fmt.Errorf("exactly one of clusterrole or role must be specified")
	}
	return nil
}
