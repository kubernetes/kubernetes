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

package rolebinding

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
)

// strategy implements behavior for RoleBindings
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// strategy is the default logic that applies when creating and updating
// RoleBinding objects.
var Strategy = strategy{api.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped is true for RoleBindings.
func (strategy) NamespaceScoped() bool {
	return true
}

// AllowCreateOnUpdate is true for RoleBindings.
func (strategy) AllowCreateOnUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (strategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	_ = obj.(*rbac.RoleBinding)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newRoleBinding := obj.(*rbac.RoleBinding)
	oldRoleBinding := old.(*rbac.RoleBinding)

	_, _ = newRoleBinding, oldRoleBinding
}

// Validate validates a new RoleBinding. Validation must check for a correct signature.
func (strategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	roleBinding := obj.(*rbac.RoleBinding)
	return validation.ValidateRoleBinding(roleBinding)
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
	_ = obj.(*rbac.RoleBinding)
}

// ValidateUpdate is the default update validation for an end user.
func (strategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	newObj := obj.(*rbac.RoleBinding)
	errorList := validation.ValidateRoleBinding(newObj)
	return append(errorList, validation.ValidateRoleBindingUpdate(newObj, old.(*rbac.RoleBinding))...)
}

// If AllowUnconditionalUpdate() is true and the object specified by
// the user does not have a resource version, then generic Update()
// populates it with the latest version. Else, it checks that the
// version specified by the user matches the version of latest etcd
// object.
func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

func (s strategy) Export(ctx genericapirequest.Context, obj runtime.Object, exact bool) error {
	return nil
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	roleBinding, ok := obj.(*rbac.RoleBinding)
	if !ok {
		return nil, nil, false, fmt.Errorf("not a RoleBinding")
	}
	return labels.Set(roleBinding.Labels), SelectableFields(roleBinding), roleBinding.Initializers != nil, nil
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// SelectableFields returns a field set that can be used for filter selection
func SelectableFields(obj *rbac.RoleBinding) fields.Set {
	return nil
}
