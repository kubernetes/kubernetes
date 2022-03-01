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

package customresourcedefinition

import (
	"context"
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// strategy implements behavior for CustomResources.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStrategy(typer runtime.ObjectTyper) strategy {
	return strategy{typer, names.SimpleNameGenerator}
}

func (strategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (strategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apiextensions.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"apiextensions.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a CustomResourceDefinition before creation.
func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	crd := obj.(*apiextensions.CustomResourceDefinition)
	crd.Status = apiextensions.CustomResourceDefinitionStatus{}
	crd.Generation = 1

	for _, v := range crd.Spec.Versions {
		if v.Storage {
			if !apiextensions.IsStoredVersion(crd, v.Name) {
				crd.Status.StoredVersions = append(crd.Status.StoredVersions, v.Name)
			}
			break
		}
	}

	dropDisabledFields(crd, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (strategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCRD := obj.(*apiextensions.CustomResourceDefinition)
	oldCRD := old.(*apiextensions.CustomResourceDefinition)
	newCRD.Status = oldCRD.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the controller has *seen*.
	if !apiequality.Semantic.DeepEqual(oldCRD.Spec, newCRD.Spec) {
		newCRD.Generation = oldCRD.Generation + 1
	}

	for _, v := range newCRD.Spec.Versions {
		if v.Storage {
			if !apiextensions.IsStoredVersion(newCRD, v.Name) {
				newCRD.Status.StoredVersions = append(newCRD.Status.StoredVersions, v.Name)
			}
			break
		}
	}

	dropDisabledFields(newCRD, oldCRD)
}

// Validate validates a new CustomResourceDefinition.
func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateCustomResourceDefinition(obj.(*apiextensions.CustomResourceDefinition))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// AllowCreateOnUpdate is false for CustomResourceDefinition; this means a POST is
// needed to create one.
func (strategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate is the default update policy for CustomResourceDefinition objects.
func (strategy) AllowUnconditionalUpdate() bool {
	return false
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
}

// ValidateUpdate is the default update validation for an end user updating status.
func (strategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCustomResourceDefinitionUpdate(obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition))
}

// WarningsOnUpdate returns warnings for the given update.
func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type statusStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func NewStatusStrategy(typer runtime.ObjectTyper) statusStrategy {
	return statusStrategy{typer, names.SimpleNameGenerator}
}

func (statusStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (statusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apiextensions.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"apiextensions.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (statusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newObj := obj.(*apiextensions.CustomResourceDefinition)
	oldObj := old.(*apiextensions.CustomResourceDefinition)
	newObj.Spec = oldObj.Spec

	// Status updates are for only for updating status, not objectmeta.
	metav1.ResetObjectMetaForStatus(&newObj.ObjectMeta, &newObj.ObjectMeta)
}

func (statusStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (statusStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (statusStrategy) Canonicalize(obj runtime.Object) {
}

func (statusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateUpdateCustomResourceDefinitionStatus(obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition))
}

// WarningsOnUpdate returns warnings for the given update.
func (statusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	apiserver, ok := obj.(*apiextensions.CustomResourceDefinition)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a CustomResourceDefinition")
	}
	return labels.Set(apiserver.ObjectMeta.Labels), CustomResourceDefinitionToSelectableFields(apiserver), nil
}

// MatchCustomResourceDefinition is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchCustomResourceDefinition(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// CustomResourceDefinitionToSelectableFields returns a field set that represents the object.
func CustomResourceDefinitionToSelectableFields(obj *apiextensions.CustomResourceDefinition) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, true)
}

// dropDisabledFields drops disabled fields that are not used if their associated feature gates
// are not enabled.
func dropDisabledFields(newCRD *apiextensions.CustomResourceDefinition, oldCRD *apiextensions.CustomResourceDefinition) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CustomResourceValidationExpressions) && (oldCRD == nil || (oldCRD != nil && !specHasXValidations(&oldCRD.Spec))) {
		if newCRD.Spec.Validation != nil {
			dropXValidationsField(newCRD.Spec.Validation.OpenAPIV3Schema)
		}
		for _, v := range newCRD.Spec.Versions {
			if v.Schema != nil {
				dropXValidationsField(v.Schema.OpenAPIV3Schema)
			}
		}
	}
}

// dropXValidationsField drops field XValidations from CRD schema
func dropXValidationsField(schema *apiextensions.JSONSchemaProps) {
	if schema == nil {
		return
	}
	schema.XValidations = nil
	if schema.AdditionalProperties != nil {
		dropXValidationsField(schema.AdditionalProperties.Schema)
	}
	for def, jsonSchema := range schema.Properties {
		dropXValidationsField(&jsonSchema)
		schema.Properties[def] = jsonSchema
	}
	if schema.Items != nil {
		dropXValidationsField(schema.Items.Schema)
		for i, jsonSchema := range schema.Items.JSONSchemas {
			dropXValidationsField(&jsonSchema)
			schema.Items.JSONSchemas[i] = jsonSchema
		}
	}
	for def, jsonSchemaPropsOrStringArray := range schema.Dependencies {
		dropXValidationsField(jsonSchemaPropsOrStringArray.Schema)
		schema.Dependencies[def] = jsonSchemaPropsOrStringArray
	}
}

func specHasXValidations(spec *apiextensions.CustomResourceDefinitionSpec) bool {
	return validation.HasSchemaWith(spec, schemaHasXValidations)
}

func schemaHasXValidations(s *apiextensions.JSONSchemaProps) bool {
	return validation.SchemaHas(s, func(s *apiextensions.JSONSchemaProps) bool {
		return s.XValidations != nil
	})
}
