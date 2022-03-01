/*
Copyright 2018 The Kubernetes Authors.

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
	"context"
	"fmt"
	"math"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kube-openapi/pkg/validation/validate"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
)

type customResourceValidator struct {
	namespaceScoped       bool
	kind                  schema.GroupVersionKind
	schemaValidator       *validate.SchemaValidator
	statusSchemaValidator *validate.SchemaValidator
}

func (a customResourceValidator) Validate(ctx context.Context, obj runtime.Object, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), u, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", u))}
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}

	if errs := a.ValidateTypeMeta(ctx, u); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessor(accessor, a.namespaceScoped, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	allErrs = append(allErrs, apiservervalidation.ValidateCustomResource(nil, u.UnstructuredContent(), a.schemaValidator)...)
	allErrs = append(allErrs, a.ValidateScaleSpec(ctx, u, scale)...)
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, u, scale)...)

	return allErrs
}

func (a customResourceValidator) ValidateUpdate(ctx context.Context, obj, old runtime.Object, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), u, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", u))}
	}
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}
	oldAccessor, err := meta.Accessor(old)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}

	if errs := a.ValidateTypeMeta(ctx, u); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessorUpdate(objAccessor, oldAccessor, field.NewPath("metadata"))...)
	allErrs = append(allErrs, apiservervalidation.ValidateCustomResource(nil, u.UnstructuredContent(), a.schemaValidator)...)
	allErrs = append(allErrs, a.ValidateScaleSpec(ctx, u, scale)...)
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, u, scale)...)

	return allErrs
}

// WarningsOnUpdate returns warnings for the given update.
func (customResourceValidator) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (a customResourceValidator) ValidateStatusUpdate(ctx context.Context, obj, old runtime.Object, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return field.ErrorList{field.Invalid(field.NewPath(""), u, fmt.Sprintf("has type %T. Must be a pointer to an Unstructured type", u))}
	}
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}
	oldAccessor, err := meta.Accessor(old)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("metadata"), nil, err.Error())}
	}

	if errs := a.ValidateTypeMeta(ctx, u); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessorUpdate(objAccessor, oldAccessor, field.NewPath("metadata"))...)
	allErrs = append(allErrs, apiservervalidation.ValidateCustomResource(nil, u.UnstructuredContent(), a.schemaValidator)...)
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, u, scale)...)

	return allErrs
}

func (a customResourceValidator) ValidateTypeMeta(ctx context.Context, obj *unstructured.Unstructured) field.ErrorList {
	typeAccessor, err := meta.TypeAccessor(obj)
	if err != nil {
		return field.ErrorList{field.Invalid(field.NewPath("kind"), nil, err.Error())}
	}

	var allErrs field.ErrorList
	if typeAccessor.GetKind() != a.kind.Kind {
		allErrs = append(allErrs, field.Invalid(field.NewPath("kind"), typeAccessor.GetKind(), fmt.Sprintf("must be %v", a.kind.Kind)))
	}
	if typeAccessor.GetAPIVersion() != a.kind.Group+"/"+a.kind.Version {
		allErrs = append(allErrs, field.Invalid(field.NewPath("apiVersion"), typeAccessor.GetAPIVersion(), fmt.Sprintf("must be %v", a.kind.Group+"/"+a.kind.Version)))
	}
	return allErrs
}

func (a customResourceValidator) ValidateScaleSpec(ctx context.Context, obj *unstructured.Unstructured, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	if scale == nil {
		return nil
	}

	var allErrs field.ErrorList

	// validate specReplicas
	specReplicasPath := strings.TrimPrefix(scale.SpecReplicasPath, ".") // ignore leading period
	specReplicas, _, err := unstructured.NestedInt64(obj.UnstructuredContent(), strings.Split(specReplicasPath, ".")...)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.SpecReplicasPath), specReplicas, err.Error()))
	} else if specReplicas < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.SpecReplicasPath), specReplicas, "should be a non-negative integer"))
	} else if specReplicas > math.MaxInt32 {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.SpecReplicasPath), specReplicas, fmt.Sprintf("should be less than or equal to %v", math.MaxInt32)))
	}

	return allErrs
}

func (a customResourceValidator) ValidateScaleStatus(ctx context.Context, obj *unstructured.Unstructured, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	if scale == nil {
		return nil
	}

	var allErrs field.ErrorList

	// validate statusReplicas
	statusReplicasPath := strings.TrimPrefix(scale.StatusReplicasPath, ".") // ignore leading period
	statusReplicas, _, err := unstructured.NestedInt64(obj.UnstructuredContent(), strings.Split(statusReplicasPath, ".")...)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.StatusReplicasPath), statusReplicas, err.Error()))
	} else if statusReplicas < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.StatusReplicasPath), statusReplicas, "should be a non-negative integer"))
	} else if statusReplicas > math.MaxInt32 {
		allErrs = append(allErrs, field.Invalid(field.NewPath(scale.StatusReplicasPath), statusReplicas, fmt.Sprintf("should be less than or equal to %v", math.MaxInt32)))
	}

	// validate labelSelector
	if scale.LabelSelectorPath != nil {
		labelSelectorPath := strings.TrimPrefix(*scale.LabelSelectorPath, ".") // ignore leading period
		labelSelector, _, err := unstructured.NestedString(obj.UnstructuredContent(), strings.Split(labelSelectorPath, ".")...)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath(*scale.LabelSelectorPath), labelSelector, err.Error()))
		}
	}

	return allErrs
}
