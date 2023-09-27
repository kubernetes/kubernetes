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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	apimachineryvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsvalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
)

type customResourceValidator struct {
	kcpValidateName validation.ValidateNameFunc

	namespaceScoped       bool
	kind                  schema.GroupVersionKind
	schemaValidator       apiextensionsvalidation.SchemaValidator
	statusSchemaValidator apiextensionsvalidation.SchemaValidator
}

func (a customResourceValidator) Validate(ctx context.Context, obj *unstructured.Unstructured, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	if errs := a.ValidateTypeMeta(ctx, obj); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessor(obj, a.namespaceScoped, a.kcpValidateName, field.NewPath("metadata"))...)
	allErrs = append(allErrs, apiextensionsvalidation.ValidateCustomResource(nil, obj.UnstructuredContent(), a.schemaValidator)...)
	allErrs = append(allErrs, a.ValidateScaleSpec(ctx, obj, scale)...)
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, obj, scale)...)

	return allErrs
}

func (a customResourceValidator) ValidateUpdate(ctx context.Context, obj, old *unstructured.Unstructured, scale *apiextensions.CustomResourceSubresourceScale, options ...apiextensionsvalidation.ValidationOption) field.ErrorList {
	if errs := a.ValidateTypeMeta(ctx, obj); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessorUpdate(obj, old, field.NewPath("metadata"))...)
	allErrs = append(allErrs, apiextensionsvalidation.ValidateCustomResourceUpdate(nil, obj.UnstructuredContent(), old.UnstructuredContent(), a.schemaValidator, options...)...)
	allErrs = append(allErrs, a.ValidateScaleSpec(ctx, obj, scale)...)
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, obj, scale)...)

	return allErrs
}

var standardFinalizers = sets.NewString(
	metav1.FinalizerOrphanDependents,
	metav1.FinalizerDeleteDependents,
	string(corev1.FinalizerKubernetes),
)

func validateKubeFinalizerName(stringValue string, fldPath *field.Path) []string {
	var allWarnings []string
	for _, msg := range apimachineryvalidation.IsQualifiedName(stringValue) {
		allWarnings = append(allWarnings, fmt.Sprintf("%s: %q: %s", fldPath.String(), stringValue, msg))
	}
	if len(strings.Split(stringValue, "/")) == 1 {
		if !standardFinalizers.Has(stringValue) {
			allWarnings = append(allWarnings, fmt.Sprintf("%s: %q: prefer a domain-qualified finalizer name to avoid accidental conflicts with other finalizer writers", fldPath.String(), stringValue))
		}
	}
	return allWarnings
}

func (a customResourceValidator) ValidateStatusUpdate(ctx context.Context, obj, old *unstructured.Unstructured, scale *apiextensions.CustomResourceSubresourceScale) field.ErrorList {
	if errs := a.ValidateTypeMeta(ctx, obj); len(errs) > 0 {
		return errs
	}

	var allErrs field.ErrorList

	allErrs = append(allErrs, validation.ValidateObjectMetaAccessorUpdate(obj, old, field.NewPath("metadata"))...)
	if status, hasStatus := obj.UnstructuredContent()["status"]; hasStatus {
		allErrs = append(allErrs, apiextensionsvalidation.ValidateCustomResourceUpdate(field.NewPath("status"), status, old.UnstructuredContent()["status"], a.statusSchemaValidator)...)
	}
	allErrs = append(allErrs, a.ValidateScaleStatus(ctx, obj, scale)...)

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
	// HACK: support the case when we add core resources through CRDs (KCP scenario)
	expectedAPIVersion := a.kind.Group + "/" + a.kind.Version
	if a.kind.Group == "" {
		expectedAPIVersion = a.kind.Version
	}
	if typeAccessor.GetAPIVersion() != expectedAPIVersion {
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
