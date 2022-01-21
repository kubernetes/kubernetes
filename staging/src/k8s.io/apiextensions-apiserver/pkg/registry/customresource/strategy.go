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
	"context"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	structurallisttype "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/listtype"
	schemaobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	apiserverstorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kube-openapi/pkg/validation/validate"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// customResourceStrategy implements behavior for CustomResources.
type customResourceStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator

	namespaceScoped   bool
	validator         customResourceValidator
	structuralSchemas map[string]*structuralschema.Structural
	celValidators     map[string]*cel.Validator
	status            *apiextensions.CustomResourceSubresourceStatus
	scale             *apiextensions.CustomResourceSubresourceScale
	kind              schema.GroupVersionKind
}

func NewStrategy(typer runtime.ObjectTyper, namespaceScoped bool, kind schema.GroupVersionKind, schemaValidator, statusSchemaValidator *validate.SchemaValidator, structuralSchemas map[string]*structuralschema.Structural, status *apiextensions.CustomResourceSubresourceStatus, scale *apiextensions.CustomResourceSubresourceScale) customResourceStrategy {
	celValidators := map[string]*cel.Validator{}
	if utilfeature.DefaultFeatureGate.Enabled(features.CustomResourceValidationExpressions) {
		for name, s := range structuralSchemas {
			v := cel.NewValidator(s) // CEL programs are compiled and cached here
			if v != nil {
				celValidators[name] = v
			}
		}
	}

	return customResourceStrategy{
		ObjectTyper:     typer,
		NameGenerator:   names.SimpleNameGenerator,
		namespaceScoped: namespaceScoped,
		status:          status,
		scale:           scale,
		validator: customResourceValidator{
			namespaceScoped:       namespaceScoped,
			kind:                  kind,
			schemaValidator:       schemaValidator,
			statusSchemaValidator: statusSchemaValidator,
		},
		structuralSchemas: structuralSchemas,
		celValidators:     celValidators,
		kind:              kind,
	}
}

func (a customResourceStrategy) NamespaceScoped() bool {
	return a.namespaceScoped
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (a customResourceStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{}

	if a.status != nil {
		fields[fieldpath.APIVersion(a.kind.GroupVersion().String())] = fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		)
	}

	return fields
}

// PrepareForCreate clears the status of a CustomResource before creation.
func (a customResourceStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	if a.status != nil {
		customResourceObject := obj.(*unstructured.Unstructured)
		customResource := customResourceObject.UnstructuredContent()

		// create cannot set status
		delete(customResource, "status")
	}

	accessor, _ := meta.Accessor(obj)
	accessor.SetGeneration(1)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (a customResourceStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCustomResourceObject := obj.(*unstructured.Unstructured)
	oldCustomResourceObject := old.(*unstructured.Unstructured)

	newCustomResource := newCustomResourceObject.UnstructuredContent()
	oldCustomResource := oldCustomResourceObject.UnstructuredContent()

	// If the /status subresource endpoint is installed, update is not allowed to set status.
	if a.status != nil {
		_, ok1 := newCustomResource["status"]
		_, ok2 := oldCustomResource["status"]
		switch {
		case ok2:
			newCustomResource["status"] = oldCustomResource["status"]
		case ok1:
			delete(newCustomResource, "status")
		}
	}

	// except for the changes to `metadata`, any other changes
	// cause the generation to increment.
	newCopyContent := copyNonMetadata(newCustomResource)
	oldCopyContent := copyNonMetadata(oldCustomResource)
	if !apiequality.Semantic.DeepEqual(newCopyContent, oldCopyContent) {
		oldAccessor, _ := meta.Accessor(oldCustomResourceObject)
		newAccessor, _ := meta.Accessor(newCustomResourceObject)
		newAccessor.SetGeneration(oldAccessor.GetGeneration() + 1)
	}
}

func copyNonMetadata(original map[string]interface{}) map[string]interface{} {
	ret := make(map[string]interface{})
	for key, val := range original {
		if key == "metadata" {
			continue
		}
		ret[key] = val
	}
	return ret
}

// Validate validates a new CustomResource.
func (a customResourceStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	var errs field.ErrorList
	errs = append(errs, a.validator.Validate(ctx, obj, a.scale)...)

	// validate embedded resources
	if u, ok := obj.(*unstructured.Unstructured); ok {
		v := obj.GetObjectKind().GroupVersionKind().Version
		errs = append(errs, schemaobjectmeta.Validate(nil, u.Object, a.structuralSchemas[v], false)...)

		// validate x-kubernetes-list-type "map" and "set" invariant
		errs = append(errs, structurallisttype.ValidateListSetsAndMaps(nil, a.structuralSchemas[v], u.Object)...)

		// validate x-kubernetes-validations rules
		if celValidator, ok := a.celValidators[v]; ok {
			errs = append(errs, celValidator.Validate(nil, a.structuralSchemas[v], u.Object)...)
		}
	}

	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (customResourceStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (customResourceStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for CustomResources; this means a POST is
// needed to create one.
func (customResourceStrategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate is the default update policy for CustomResource objects.
func (customResourceStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user updating status.
func (a customResourceStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var errs field.ErrorList
	errs = append(errs, a.validator.ValidateUpdate(ctx, obj, old, a.scale)...)

	uNew, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return errs
	}
	uOld, ok := old.(*unstructured.Unstructured)
	if !ok {
		return errs
	}

	// Checks the embedded objects. We don't make a difference between update and create for those.
	v := obj.GetObjectKind().GroupVersionKind().Version
	errs = append(errs, schemaobjectmeta.Validate(nil, uNew.Object, a.structuralSchemas[v], false)...)

	// ratcheting validation of x-kubernetes-list-type value map and set
	if oldErrs := structurallisttype.ValidateListSetsAndMaps(nil, a.structuralSchemas[v], uOld.Object); len(oldErrs) == 0 {
		errs = append(errs, structurallisttype.ValidateListSetsAndMaps(nil, a.structuralSchemas[v], uNew.Object)...)
	}

	// validate x-kubernetes-validations rules
	if celValidator, ok := a.celValidators[v]; ok {
		errs = append(errs, celValidator.Validate(nil, a.structuralSchemas[v], uNew.Object)...)
	}

	return errs
}

// WarningsOnUpdate returns warnings for the given update.
func (customResourceStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func (a customResourceStrategy) GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, err
	}
	return labels.Set(accessor.GetLabels()), objectMetaFieldsSet(accessor, a.namespaceScoped), nil
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

// MatchCustomResourceDefinitionStorage is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func (a customResourceStrategy) MatchCustomResourceDefinitionStorage(label labels.Selector, field fields.Selector) apiserverstorage.SelectionPredicate {
	return apiserverstorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: a.GetAttrs,
	}
}
