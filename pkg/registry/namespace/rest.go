/*
Copyright 2014 Google Inc. All rights reserved.

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

package namespace

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// namespaceStrategy implements behavior for Namespaces
type namespaceStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Namespace
// objects via the REST API.
var Strategy = namespaceStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for namespaces.
func (namespaceStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (namespaceStrategy) ResetBeforeCreate(obj runtime.Object) {
	namespace := obj.(*api.Namespace)
	namespace.Status = api.NamespaceStatus{
		Phase: api.NamespaceActive,
	}
}

// Validate validates a new namespace.
func (namespaceStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	namespace := obj.(*api.Namespace)
	return validation.ValidateNamespace(namespace)
}

// AllowCreateOnUpdate is false for namespaces.
func (namespaceStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (namespaceStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	return validation.ValidateNamespaceUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

type namespaceStatusStrategy struct {
	namespaceStrategy
}

var StatusStrategy = namespaceStatusStrategy{Strategy}

func (namespaceStatusStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	// TODO: merge valid fields after update
	return validation.ValidateNamespaceStatusUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

type namespaceFinalizeStrategy struct {
	namespaceStrategy
}

var FinalizeStrategy = namespaceFinalizeStrategy{Strategy}

func (namespaceFinalizeStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	return validation.ValidateNamespaceFinalizeUpdate(obj.(*api.Namespace), old.(*api.Namespace))
}

// MatchNamespace returns a generic matcher for a given label and field selector.
func MatchNamespace(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		namespaceObj, ok := obj.(*api.Namespace)
		if !ok {
			return false, fmt.Errorf("not a namespace")
		}
		fields := NamespaceToSelectableFields(namespaceObj)
		return label.Matches(labels.Set(namespaceObj.Labels)) && field.Matches(fields), nil
	})
}

// NamespaceToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func NamespaceToSelectableFields(namespace *api.Namespace) labels.Set {
	return labels.Set{
		"name":         namespace.Name,
		"status.phase": string(namespace.Status.Phase),
	}
}
