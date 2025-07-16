/*
Copyright 2014 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replicationcontroller

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// rcStrategy implements verification logic for Replication Controllers.
type rcStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = rcStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents for v1 for backwards compatibility,
// and DeleteDependents for all other versions.
func (rcStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	switch groupVersion {
	case corev1.SchemeGroupVersion:
		// for back compatibility
		return rest.OrphanDependents
	default:
		return rest.DeleteDependents
	}
}

// NamespaceScoped returns true because all Replication Controllers need to be within a namespace.
func (rcStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (rcStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a replication controller before creation.
func (rcStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	controller := obj.(*api.ReplicationController)
	controller.Status = api.ReplicationControllerStatus{}

	controller.Generation = 1

	pod.DropDisabledTemplateFields(controller.Spec.Template, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (rcStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newController := obj.(*api.ReplicationController)
	oldController := old.(*api.ReplicationController)
	// update is not allowed to set status
	newController.Status = oldController.Status

	pod.DropDisabledTemplateFields(newController.Spec.Template, oldController.Spec.Template)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the controller has *seen*.
	if !apiequality.Semantic.DeepEqual(oldController.Spec, newController.Spec) {
		newController.Generation = oldController.Generation + 1
	}
}

// Validate validates a new replication controller.
func (rcStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	controller := obj.(*api.ReplicationController)
	opts := pod.GetValidationOptionsFromPodTemplate(controller.Spec.Template, nil)

	// Run imperative validation
	allErrs := corevalidation.ValidateReplicationController(controller, opts)

	// If DeclarativeValidation feature gate is enabled, also run declarative validation
	// FIXME: isSpecRequest(ctx) limits Declarative validation to the spec until subresource support is introduced.
	if utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidation) && isSpecRequest(ctx) {
		// Determine if takeover is enabled
		takeover := utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidationTakeover)

		// Run declarative validation with panic recovery
		declarativeErrs := rest.ValidateDeclarativelyWithRecovery(ctx, nil, legacyscheme.Scheme, controller, takeover)

		// Compare imperative and declarative errors and log + emit metric if there's a mismatch
		rest.CompareDeclarativeErrorsAndEmitMismatches(ctx, allErrs, declarativeErrs, takeover)

		// Only apply declarative errors if takeover is enabled
		if takeover {
			allErrs = append(allErrs.RemoveCoveredByDeclarative(), declarativeErrs...)
		}
	}
	return allErrs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (rcStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newRC := obj.(*api.ReplicationController)
	var warnings []string
	if msgs := utilvalidation.IsDNS1123Label(newRC.Name); len(msgs) != 0 {
		warnings = append(warnings, fmt.Sprintf("metadata.name: this is used in Pod names and hostnames, which can result in surprising behavior; a DNS label is recommended: %v", msgs))
	}
	warnings = append(warnings, pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), newRC.Spec.Template, nil)...)
	return warnings
}

// Canonicalize normalizes the object after validation.
func (rcStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for replication controllers; this means a POST is
// needed to create one.
func (rcStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (rcStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	oldRc := old.(*api.ReplicationController)
	newRc := obj.(*api.ReplicationController)

	opts := pod.GetValidationOptionsFromPodTemplate(newRc.Spec.Template, oldRc.Spec.Template)
	// This should be fixed to avoid the redundant calls, but carefully.
	validationErrorList := corevalidation.ValidateReplicationController(newRc, opts)
	updateErrorList := corevalidation.ValidateReplicationControllerUpdate(newRc, oldRc, opts)
	errs := append(validationErrorList, updateErrorList...)

	for key, value := range helper.NonConvertibleFields(oldRc.Annotations) {
		parts := strings.Split(key, "/")
		if len(parts) != 2 {
			continue
		}
		brokenField := parts[1]

		switch {
		case strings.Contains(brokenField, "selector"):
			if !apiequality.Semantic.DeepEqual(oldRc.Spec.Selector, newRc.Spec.Selector) {
				errs = append(errs, field.Invalid(field.NewPath("spec").Child("selector"), newRc.Spec.Selector, "cannot update non-convertible selector"))
			}
		default:
			errs = append(errs, &field.Error{Type: field.ErrorTypeNotFound, BadValue: value, Field: brokenField, Detail: "unknown non-convertible field"})
		}
	}

	// If DeclarativeValidation feature gate is enabled, also run declarative validation
	// FIXME: This limits Declarative validation to the spec until subresource support is introduced.
	if utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidation) && isSpecRequest(ctx) {
		// Determine if takeover is enabled
		takeover := utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidationTakeover)

		// Run declarative update validation with panic recovery
		declarativeErrs := rest.ValidateUpdateDeclarativelyWithRecovery(ctx, nil, legacyscheme.Scheme, newRc, oldRc, takeover)

		// Compare imperative and declarative errors and emit metric if there's a mismatch
		rest.CompareDeclarativeErrorsAndEmitMismatches(ctx, errs, declarativeErrs, takeover)

		// Only apply declarative errors if takeover is enabled
		if takeover {
			errs = append(errs.RemoveCoveredByDeclarative(), declarativeErrs...)
		}
	}

	return errs
}

func isSpecRequest(ctx context.Context) bool {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		return len(requestInfo.Subresource) == 0
	}
	return false
}

// WarningsOnUpdate returns warnings for the given update.
func (rcStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	oldRc := old.(*api.ReplicationController)
	newRc := obj.(*api.ReplicationController)
	if oldRc.Generation != newRc.Generation {
		warnings = pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), oldRc.Spec.Template, newRc.Spec.Template)
	}
	return warnings
}

func (rcStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// ControllerToSelectableFields returns a field set that represents the object.
func ControllerToSelectableFields(controller *api.ReplicationController) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&controller.ObjectMeta, true)
	controllerSpecificFieldsSet := fields.Set{
		"status.replicas": strconv.Itoa(int(controller.Status.Replicas)),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, controllerSpecificFieldsSet)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	rc, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a replication controller")
	}
	return labels.Set(rc.ObjectMeta.Labels), ControllerToSelectableFields(rc), nil
}

// MatchController is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchController(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

type rcStatusStrategy struct {
	rcStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = rcStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (rcStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
}

func (rcStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRc := obj.(*api.ReplicationController)
	oldRc := old.(*api.ReplicationController)
	// update is not allowed to set spec
	newRc.Spec = oldRc.Spec
}

func (rcStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return corevalidation.ValidateReplicationControllerStatusUpdate(obj.(*api.ReplicationController), old.(*api.ReplicationController))
}

// WarningsOnUpdate returns warnings for the given update.
func (rcStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
