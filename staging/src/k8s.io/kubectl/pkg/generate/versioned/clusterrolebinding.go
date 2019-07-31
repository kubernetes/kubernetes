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

	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubectl/pkg/generate"
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
var _ generate.Generator = &ClusterRoleBindingGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ generate.StructuredGenerator = &ClusterRoleBindingGeneratorV1{}

// Generate returns a clusterRoleBinding using the specified parameters.
func (s ClusterRoleBindingGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := generate.ValidateParams(s.ParamNames(), genericParams)
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
func (s ClusterRoleBindingGeneratorV1) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "name", Required: true},
		{Name: "clusterrole", Required: false},
		{Name: "user", Required: false},
		{Name: "group", Required: false},
		{Name: "serviceaccount", Required: false},
	}
}

// StructuredGenerate outputs a clusterRoleBinding object using the configured fields.
func (s ClusterRoleBindingGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	clusterRoleBinding := &rbacv1beta1.ClusterRoleBinding{}
	clusterRoleBinding.Name = s.Name
	clusterRoleBinding.RoleRef = rbacv1beta1.RoleRef{
		APIGroup: rbacv1beta1.GroupName,
		Kind:     "ClusterRole",
		Name:     s.ClusterRole,
	}
	for _, user := range sets.NewString(s.Users...).List() {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1beta1.Subject{
			Kind:     rbacv1beta1.UserKind,
			APIGroup: rbacv1beta1.GroupName,
			Name:     user,
		})
	}
	for _, group := range sets.NewString(s.Groups...).List() {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1beta1.Subject{
			Kind:     rbacv1beta1.GroupKind,
			APIGroup: rbacv1beta1.GroupName,
			Name:     group,
		})
	}
	for _, sa := range sets.NewString(s.ServiceAccounts...).List() {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[0] == "" || tokens[1] == "" {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1beta1.Subject{
			Kind:      rbacv1beta1.ServiceAccountKind,
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
