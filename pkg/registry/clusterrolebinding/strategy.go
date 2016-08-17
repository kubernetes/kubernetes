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

package clusterrolebinding

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

// strategy implements behavior for ClusterRoleBindings
type strategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// strategy is the default logic that applies when creating and updating
// ClusterRoleBinding objects.
var Strategy = strategy{api.Scheme, api.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped is true for ClusterRoleBindings.
func (strategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate is true for ClusterRoleBindings.
func (strategy) AllowCreateOnUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (strategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	_ = obj.(*rbac.ClusterRoleBinding)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newClusterRoleBinding := obj.(*rbac.ClusterRoleBinding)
	oldClusterRoleBinding := old.(*rbac.ClusterRoleBinding)

	_, _ = newClusterRoleBinding, oldClusterRoleBinding
}

// Validate validates a new ClusterRoleBinding. Validation must check for a correct signature.
func (strategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	clusterRoleBinding := obj.(*rbac.ClusterRoleBinding)
	return validation.ValidateClusterRoleBinding(clusterRoleBinding)
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
	_ = obj.(*rbac.ClusterRoleBinding)
}

// ValidateUpdate is the default update validation for an end user.
func (strategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	newObj := obj.(*rbac.ClusterRoleBinding)
	errorList := validation.ValidateClusterRoleBinding(newObj)
	return append(errorList, validation.ValidateClusterRoleBindingUpdate(newObj, old.(*rbac.ClusterRoleBinding))...)
}

// If AllowUnconditionalUpdate() is true and the object specified by
// the user does not have a resource version, then generic Update()
// populates it with the latest version. Else, it checks that the
// version specified by the user matches the version of latest etcd
// object.
func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

func (s strategy) Export(ctx api.Context, obj runtime.Object, exact bool) error {
	return nil
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		sa, ok := obj.(*rbac.ClusterRoleBinding)
		if !ok {
			return false, fmt.Errorf("not a ClusterRoleBinding")
		}
		fields := SelectableFields(sa)
		return label.Matches(labels.Set(sa.Labels)) && field.Matches(fields), nil
	})
}

// SelectableFields returns a label set that can be used for filter selection
func SelectableFields(obj *rbac.ClusterRoleBinding) labels.Set {
	return labels.Set{}
}
