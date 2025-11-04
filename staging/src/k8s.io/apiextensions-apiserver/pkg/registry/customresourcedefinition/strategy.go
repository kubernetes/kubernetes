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

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kube-openapi/pkg/validation/spec"
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
	return validation.ValidateCustomResourceDefinition(ctx, obj.(*apiextensions.CustomResourceDefinition))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newCRD := obj.(*apiextensions.CustomResourceDefinition)

	var warnings []string

	// Use the system's default compatibility version for format checking
	compatibilityVersion := environment.DefaultCompatibilityVersion()

	// Get unrecognized formats from the new CRD
	unrecognizedFormats := getUnrecognizedFormatsInCRD(&newCRD.Spec, compatibilityVersion)

	// Create warnings for unrecognized formats
	for _, format := range unrecognizedFormats {
		warnings = append(warnings, fmt.Sprintf("unrecognized format %q", format))
	}

	return warnings
}

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
	return validation.ValidateCustomResourceDefinitionUpdate(ctx, obj.(*apiextensions.CustomResourceDefinition), old.(*apiextensions.CustomResourceDefinition))
}

// WarningsOnUpdate returns warnings for the given update.
func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	newCRD := obj.(*apiextensions.CustomResourceDefinition)
	oldCRD := old.(*apiextensions.CustomResourceDefinition)

	var warnings []string

	// Use the system's default compatibility version for format checking
	compatibilityVersion := environment.DefaultCompatibilityVersion()

	// Get unrecognized formats from both old and new CRDs
	oldUnrecognizedFormats := getUnrecognizedFormatsInCRD(&oldCRD.Spec, compatibilityVersion)
	newUnrecognizedFormats := getUnrecognizedFormatsInCRD(&newCRD.Spec, compatibilityVersion)

	// Find newly introduced unrecognized formats (ratcheting)
	oldFormatsSet := make(map[string]bool)
	for _, format := range oldUnrecognizedFormats {
		oldFormatsSet[format] = true
	}

	var newlyIntroducedFormats []string
	for _, format := range newUnrecognizedFormats {
		if !oldFormatsSet[format] {
			newlyIntroducedFormats = append(newlyIntroducedFormats, format)
		}
	}

	// Create warnings for newly introduced unrecognized formats
	for _, format := range newlyIntroducedFormats {
		warnings = append(warnings, fmt.Sprintf("unrecognized format %q", format))
	}

	return warnings
}

// getUnrecognizedFormatsInCRD returns a list of unrecognized formats found in the CRD schema.
func getUnrecognizedFormatsInCRD(spec *apiextensions.CustomResourceDefinitionSpec, compatibilityVersion *version.Version) []string {
	var unrecognizedFormats []string

	// Check top-level validation schema (deprecated in v1, but still supported for backward compatibility)
	if spec.Validation != nil && spec.Validation.OpenAPIV3Schema != nil {
		unrecognizedFormats = append(unrecognizedFormats, getUnrecognizedFormatsInSchema(spec.Validation.OpenAPIV3Schema, compatibilityVersion)...)
	}

	// Check per-version schemas
	for _, v := range spec.Versions {
		if v.Schema != nil && v.Schema.OpenAPIV3Schema != nil {
			unrecognizedFormats = append(unrecognizedFormats, getUnrecognizedFormatsInSchema(v.Schema.OpenAPIV3Schema, compatibilityVersion)...)
		}
	}

	return unrecognizedFormats
}

// getUnrecognizedFormatsInSchema recursively traverses the schema and collects unrecognized formats.
func getUnrecognizedFormatsInSchema(schema *apiextensions.JSONSchemaProps, compatibilityVersion *version.Version) []string {
	var unrecognizedFormats []string

	// Use the existing SchemaHas function to traverse the schema
	validation.SchemaHas(schema, func(s *apiextensions.JSONSchemaProps) bool {
		if len(s.Format) > 0 {
			// Convert to spec.Schema for format validation
			specSchema := &spec.Schema{SchemaProps: spec.SchemaProps{Format: s.Format, Type: []string{s.Type}}}
			if formats := apiservervalidation.GetUnrecognizedFormats(specSchema, compatibilityVersion); len(formats) > 0 {
				unrecognizedFormats = append(unrecognizedFormats, formats...)
			}
		}
		return false // Continue traversing
	})

	return unrecognizedFormats
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
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CRDValidationRatcheting) && (oldCRD == nil || (oldCRD != nil && !specHasOptionalOldSelf(&oldCRD.Spec))) {
		if newCRD.Spec.Validation != nil {
			dropOptionalOldSelfField(newCRD.Spec.Validation.OpenAPIV3Schema)
		}

		for _, v := range newCRD.Spec.Versions {
			if v.Schema != nil {
				dropOptionalOldSelfField(v.Schema.OpenAPIV3Schema)
			}
		}
	}
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceFieldSelectors) && (oldCRD == nil || (oldCRD != nil && !specHasSelectableFields(&oldCRD.Spec))) {
		dropSelectableFields(&newCRD.Spec)
	}
	if !utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CRDObservedGenerationTracking) && (oldCRD == nil || !observedGenerationTrackingInUse(&oldCRD.Status)) {
		dropObservedGeneration(&newCRD.Status)
	}
}

// dropOptionalOldSelfField drops field optionalOldSelf from CRD schema
func dropOptionalOldSelfField(schema *apiextensions.JSONSchemaProps) {
	if schema == nil {
		return
	}
	for i := range schema.XValidations {
		schema.XValidations[i].OptionalOldSelf = nil
	}

	if schema.AdditionalProperties != nil {
		dropOptionalOldSelfField(schema.AdditionalProperties.Schema)
	}
	for def, jsonSchema := range schema.Properties {
		dropOptionalOldSelfField(&jsonSchema)
		schema.Properties[def] = jsonSchema
	}
	if schema.Items != nil {
		dropOptionalOldSelfField(schema.Items.Schema)
		for i, jsonSchema := range schema.Items.JSONSchemas {
			dropOptionalOldSelfField(&jsonSchema)
			schema.Items.JSONSchemas[i] = jsonSchema
		}
	}
}

func specHasOptionalOldSelf(spec *apiextensions.CustomResourceDefinitionSpec) bool {
	return validation.HasSchemaWith(spec, schemaHasOptionalOldSelf)
}

func schemaHasOptionalOldSelf(s *apiextensions.JSONSchemaProps) bool {
	return validation.SchemaHas(s, func(s *apiextensions.JSONSchemaProps) bool {
		for _, v := range s.XValidations {
			if v.OptionalOldSelf != nil {
				return true
			}

		}
		return false
	})
}

func dropSelectableFields(spec *apiextensions.CustomResourceDefinitionSpec) {
	spec.SelectableFields = nil
	for i := range spec.Versions {
		spec.Versions[i].SelectableFields = nil
	}
}

func dropObservedGeneration(status *apiextensions.CustomResourceDefinitionStatus) {
	status.ObservedGeneration = 0
	for i := range status.Conditions {
		status.Conditions[i].ObservedGeneration = 0
	}
}

func observedGenerationTrackingInUse(status *apiextensions.CustomResourceDefinitionStatus) bool {
	if status.ObservedGeneration != 0 {
		return true
	}
	for i := range status.Conditions {
		if status.Conditions[i].ObservedGeneration != 0 {
			return true
		}
	}
	return false
}

func specHasSelectableFields(spec *apiextensions.CustomResourceDefinitionSpec) bool {
	if spec.SelectableFields != nil {
		return true
	}
	for _, v := range spec.Versions {
		if v.SelectableFields != nil {
			return true
		}
	}

	return false
}
