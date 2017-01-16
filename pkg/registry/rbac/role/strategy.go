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

package role

import (
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/fields"
	apistorage "k8s.io/kubernetes/pkg/storage"
)

// strategy implements behavior for Roles
type strategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// strategy is the default logic that applies when creating and updating
// Role objects.
var Strategy = strategy{api.Scheme, api.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped is true for Roles.
func (strategy) NamespaceScoped() bool {
	return true
}

// AllowCreateOnUpdate is true for Roles.
func (strategy) AllowCreateOnUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (strategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	_ = obj.(*rbac.Role)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newRole := obj.(*rbac.Role)
	oldRole := old.(*rbac.Role)

	_, _ = newRole, oldRole
}

// Validate validates a new Role. Validation must check for a correct signature.
func (strategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	role := obj.(*rbac.Role)
	return validation.ValidateRole(role)
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
	_ = obj.(*rbac.Role)
}

// ValidateUpdate is the default update validation for an end user.
func (strategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	newObj := obj.(*rbac.Role)
	errorList := validation.ValidateRole(newObj)
	return append(errorList, validation.ValidateRoleUpdate(newObj, old.(*rbac.Role))...)
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
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	role, ok := obj.(*rbac.Role)
	if !ok {
		return nil, nil, fmt.Errorf("not a Role")
	}
	return labels.Set(role.Labels), SelectableFields(role), nil
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
func SelectableFields(obj *rbac.Role) fields.Set {
	return nil
}
