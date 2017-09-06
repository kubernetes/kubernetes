/*
Copyright 2017 The Kubernetes Authors.

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

package customresource

import (
	"fmt"

	"github.com/go-openapi/validate"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"

	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
)

type customResourceDefinitionStorageStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator

	namespaceScoped bool
	validator       customResourceValidator
}

func NewStrategy(typer runtime.ObjectTyper, namespaceScoped bool, kind schema.GroupVersionKind, validator *validate.SchemaValidator) customResourceDefinitionStorageStrategy {
	return customResourceDefinitionStorageStrategy{
		ObjectTyper:     typer,
		NameGenerator:   names.SimpleNameGenerator,
		namespaceScoped: namespaceScoped,
		validator: customResourceValidator{
			namespaceScoped: namespaceScoped,
			kind:            kind,
			validator:       validator,
		},
	}
}

func (a customResourceDefinitionStorageStrategy) NamespaceScoped() bool {
	return a.namespaceScoped
}

func (customResourceDefinitionStorageStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
}

func (customResourceDefinitionStorageStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
}

func (a customResourceDefinitionStorageStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	return a.validator.Validate(ctx, obj)
}

func (customResourceDefinitionStorageStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (customResourceDefinitionStorageStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (customResourceDefinitionStorageStrategy) Canonicalize(obj runtime.Object) {
}

func (a customResourceDefinitionStorageStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return a.validator.ValidateUpdate(ctx, obj, old)
}

func (a customResourceDefinitionStorageStrategy) GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, false, err
	}
	return labels.Set(accessor.GetLabels()), objectMetaFieldsSet(accessor, a.namespaceScoped), accessor.GetInitializers() != nil, nil
}

// objectMetaFieldsSet returns a fields that represent the ObjectMeta.
func objectMetaFieldsSet(objectMeta metav1.Object, namespaceScoped bool) fields.Set {
	if namespaceScoped {
		return fields.Set{
			"metadata.name":      objectMeta.GetName(),
			"metadata.namespace": objectMeta.GetNamespace(),
		}
	}
	return fields.Set{
		"metadata.name": objectMeta.GetName(),
	}
}

func (a customResourceDefinitionStorageStrategy) MatchCustomResourceDefinitionStorage(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: a.GetAttrs,
	}
}

type customResourceValidator struct {
	namespaceScoped bool
	kind            schema.GroupVersionKind
	validator       *validate.SchemaValidator
}

func (a customResourceValidator) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("kind"), nil, err.Error())}
	}
	if typeAccessor.GetKind() != a.kind.Kind {
		return field.ErrorList{field.Invalid(field.NewPath("kind"), typeAccessor.GetKind(), fmt.Sprintf("must be %v", a.kind.Kind))}
	}
	if typeAccessor.GetAPIVersion() != a.kind.Group+"/"+a.kind.Version {
		return field.ErrorList{field.Invalid(field.NewPath("apiVersion"), typeAccessor.GetKind(), fmt.Sprintf("must be %v", a.kind.Group+"/"+a.kind.Version))}
	}

	customResourceObject, ok := obj.(*unstructured.Unstructured)
	// this will never happen.
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), customResourceObject, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", customResourceObject))}
	}

	customResource := customResourceObject.UnstructuredContent()
	if err = apiservervalidation.ValidateCustomResource(customResource, a.validator); err != nil {
		return field.ErrorList{field.Invalid(field.NewPath(""), customResource, err.Error())}
	}

	return validation.ValidateObjectMetaAccessor(accessor, a.namespaceScoped, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
}

func (a customResourceValidator) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}
	oldAccessor, err := meta.Accessor(old)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("kind"), nil, err.Error())}
	}
	if typeAccessor.GetKind() != a.kind.Kind {
		return field.ErrorList{field.Invalid(field.NewPath("kind"), typeAccessor.GetKind(), fmt.Sprintf("must be %v", a.kind.Kind))}
	}
	if typeAccessor.GetAPIVersion() != a.kind.Group+"/"+a.kind.Version {
		return field.ErrorList{field.Invalid(field.NewPath("apiVersion"), typeAccessor.GetKind(), fmt.Sprintf("must be %v", a.kind.Group+"/"+a.kind.Version))}
	}

	customResourceObject, ok := obj.(*unstructured.Unstructured)
	// this will never happen.
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), customResourceObject, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", customResourceObject))}
	}

	customResource := customResourceObject.UnstructuredContent()
	if err = apiservervalidation.ValidateCustomResource(customResource, a.validator); err != nil {
		return field.ErrorList{field.Invalid(field.NewPath(""), customResource, err.Error())}
	}

	return validation.ValidateObjectMetaAccessorUpdate(objAccessor, oldAccessor, field.NewPath("metadata"))
}
