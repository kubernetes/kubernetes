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

package versioned

import (
	"fmt"

	"strings"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubectl/pkg/generate"
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
var _ generate.Generator = &RoleBindingGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ generate.StructuredGenerator = &RoleBindingGeneratorV1{}

// Generate returns a roleBinding using the specified parameters.
func (s RoleBindingGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := generate.ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &RoleBindingGeneratorV1{}
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
	delegate.Role = params["role"]
	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s RoleBindingGeneratorV1) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "name", Required: true},
		{Name: "clusterrole", Required: false},
		{Name: "role", Required: false},
		{Name: "user", Required: false},
		{Name: "group", Required: false},
		{Name: "serviceaccount", Required: false},
	}
}

// StructuredGenerate outputs a roleBinding object using the configured fields.
func (s RoleBindingGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	roleBinding := &rbacv1.RoleBinding{}
	roleBinding.Name = s.Name

	switch {
	case len(s.Role) > 0:
		roleBinding.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     s.Role,
		}
	case len(s.ClusterRole) > 0:
		roleBinding.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     s.ClusterRole,
		}
	}

	for _, user := range sets.NewString(s.Users...).List() {
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.UserKind,
			APIGroup: rbacv1.GroupName,
			Name:     user,
		})
	}
	for _, group := range sets.NewString(s.Groups...).List() {
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.GroupKind,
			APIGroup: rbacv1.GroupName,
			Name:     group,
		})
	}
	for _, sa := range sets.NewString(s.ServiceAccounts...).List() {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[1] == "" {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
			Kind:      rbacv1.ServiceAccountKind,
			APIGroup:  "",
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
