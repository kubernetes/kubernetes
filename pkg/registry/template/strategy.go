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

package template

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// templateStrategy implements behavior for Templates
type templateStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Template
// objects via the REST API.
var Strategy = templateStrategy{api.Scheme, api.SimpleNameGenerator}

var _ rest.NamespaceScopedStrategy = Strategy
var _ rest.RESTCreateStrategy = Strategy
var _ rest.RESTDeleteStrategy = Strategy
var _ rest.RESTUpdateStrategy = Strategy

//var _ rest.RESTExportStrategy = Strategy

// NamespaceScoped is true for templates.
func (templateStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (templateStrategy) PrepareForUpdate(obj, old runtime.Object) {
	// TODO: write this
}

// Canonicalize normalizes the object after validation.
func (templateStrategy) Canonicalize(obj runtime.Object) {
	// TODO: write this
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (templateStrategy) PrepareForCreate(obj runtime.Object) {
	// TODO: write this
}

// Validate validates a new template.
func (templateStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	// TODO: write this
	return field.ErrorList{}
}

// AllowCreateOnUpdate is false for templates.
func (templateStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (templateStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (templateStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	// TODO: write this
	return field.ErrorList{}
}

// templateToSelectableFields returns a label set that represents the object
// changes to the returned keys require registering conversions for existing versions using Scheme.AddFieldLabelConversionFunc
func templateToSelectableFields(template *extensions.Template) fields.Set {
	return fields.Set{
		"metadata.name": template.Name,
	}
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			template, ok := obj.(*extensions.Template)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not a deployment.")
			}
			return labels.Set(template.ObjectMeta.Labels), templateToSelectableFields(template), nil
		},
	}
}
