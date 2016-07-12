/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package clusterprotectedattribute

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type strategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default strategy for creation and update of
// ProtectedAttribute objects.
var Strategy = strategy{api.Scheme, api.SimpleNameGenerator}

var _ rest.RESTCreateStrategy = Strategy
var _ rest.RESTUpdateStrategy = Strategy

func (strategy) NamespaceScoped() bool {
	return false
}

func (strategy) AllowCreateOnUpdate() bool {
	return true
}

func (strategy) PrepareForCreate(obj runtime.Object)      {}
func (strategy) PrepareForUpdate(obj, old runtime.Object) {}
func (strategy) Canonicalize(obj runtime.Object)          {}

func (strategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateClusterProtectedAttribute(obj.(*rbac.ClusterProtectedAttribute))
}

func (strategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	newObj := obj.(*rbac.ClusterProtectedAttribute)
	oldObj := old.(*rbac.ClusterProtectedAttribute)

	errorList := append(
		validation.ValidateClusterProtectedAttribute(newObj),
		validation.ValidateClusterProtectedAttributeUpdate(newObj, oldObj)...,
	)

	return errorList
}

func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

// Matcher returns a generic matcher for a given label and field
// selector.
func Matcher(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		cpa, ok := obj.(*rbac.ClusterProtectedAttribute)
		if !ok {
			return false, fmt.Errorf("not a ClusterProtectedAttribute")
		}
		return label.Matches(labels.Set(cpa.Labels)), nil
	})
}
