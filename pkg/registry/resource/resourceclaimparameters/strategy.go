/*
Copyright 2022 The Kubernetes Authors.

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

package resourceclaimparameters

import (
	"context"
	"errors"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
)

// resourceClaimParametersStrategy implements behavior for ResourceClaimParameters objects
type resourceClaimParametersStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var Strategy = resourceClaimParametersStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (resourceClaimParametersStrategy) NamespaceScoped() bool {
	return true
}

func (resourceClaimParametersStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (resourceClaimParametersStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourceClaimParameters := obj.(*resource.ResourceClaimParameters)
	return validation.ValidateResourceClaimParameters(resourceClaimParameters)
}

func (resourceClaimParametersStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (resourceClaimParametersStrategy) Canonicalize(obj runtime.Object) {
}

func (resourceClaimParametersStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (resourceClaimParametersStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (resourceClaimParametersStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateResourceClaimParametersUpdate(obj.(*resource.ResourceClaimParameters), old.(*resource.ResourceClaimParameters))
}

func (resourceClaimParametersStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourceClaimParametersStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// Match returns a generic matcher for a given label and field selector.
func Match(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	parameters, ok := obj.(*resource.ResourceClaimParameters)
	if !ok {
		return nil, nil, errors.New("not a resourceclaim")
	}
	return labels.Set(parameters.Labels), toSelectableFields(parameters), nil
}

// toSelectableFields returns a field set that represents the object
func toSelectableFields(claim *resource.ResourceClaimParameters) fields.Set {
	fields := generic.ObjectMetaFieldsSet(&claim.ObjectMeta, true)
	return fields
}
