/*
Copyright 2015 The Kubernetes Authors.

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

package horizontalpodautoscaler

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// autoscalerStrategy implements behavior for HorizontalPodAutoscalers
type autoscalerStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating HorizontalPodAutoscaler
// objects via the REST API.
var Strategy = autoscalerStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for autoscaler.
func (autoscalerStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (autoscalerStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"autoscaling/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"autoscaling/v2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"autoscaling/v2beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"autoscaling/v2beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (autoscalerStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	newHPA := obj.(*autoscaling.HorizontalPodAutoscaler)

	// create cannot set status
	newHPA.Status = autoscaling.HorizontalPodAutoscalerStatus{}

	dropDisabledFields(newHPA, nil)
}

// Validate validates a new autoscaler.
func (autoscalerStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	autoscaler := obj.(*autoscaling.HorizontalPodAutoscaler)
	opts := validationOptionsForHorizontalPodAutoscaler(autoscaler, nil)
	return validation.ValidateHorizontalPodAutoscaler(autoscaler, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (autoscalerStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (autoscalerStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for autoscalers.
func (autoscalerStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (autoscalerStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newHPA := obj.(*autoscaling.HorizontalPodAutoscaler)
	oldHPA := old.(*autoscaling.HorizontalPodAutoscaler)
	// Update is not allowed to set status
	newHPA.Status = oldHPA.Status

	dropDisabledFields(newHPA, oldHPA)
}

// ValidateUpdate is the default update validation for an end user.
func (autoscalerStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newHPA := obj.(*autoscaling.HorizontalPodAutoscaler)
	oldHPA := old.(*autoscaling.HorizontalPodAutoscaler)
	opts := validationOptionsForHorizontalPodAutoscaler(newHPA, oldHPA)
	return validation.ValidateHorizontalPodAutoscalerUpdate(newHPA, oldHPA, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (autoscalerStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (autoscalerStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type autoscalerStatusStrategy struct {
	autoscalerStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = autoscalerStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (autoscalerStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"autoscaling/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"autoscaling/v2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"autoscaling/v2beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"autoscaling/v2beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (autoscalerStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newAutoscaler := obj.(*autoscaling.HorizontalPodAutoscaler)
	oldAutoscaler := old.(*autoscaling.HorizontalPodAutoscaler)
	// status changes are not allowed to update spec
	newAutoscaler.Spec = oldAutoscaler.Spec
}

func (autoscalerStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateHorizontalPodAutoscalerStatusUpdate(obj.(*autoscaling.HorizontalPodAutoscaler), old.(*autoscaling.HorizontalPodAutoscaler))
}

// WarningsOnUpdate returns warnings for the given update.
func (autoscalerStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func validationOptionsForHorizontalPodAutoscaler(newHPA, oldHPA *autoscaling.HorizontalPodAutoscaler) validation.HorizontalPodAutoscalerSpecValidationOptions {
	opts := validation.HorizontalPodAutoscalerSpecValidationOptions{
		MinReplicasLowerBound:           1,
		ScaleTargetRefValidationOptions: validation.CrossVersionObjectReferenceValidationOptions{AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false},
		ObjectMetricsValidationOptions: validation.CrossVersionObjectReferenceValidationOptions{
			AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
		},
	}

	oldHasZeroMinReplicas := oldHPA != nil && (oldHPA.Spec.MinReplicas != nil && *oldHPA.Spec.MinReplicas == 0)
	if utilfeature.DefaultFeatureGate.Enabled(features.HPAScaleToZero) || oldHasZeroMinReplicas {
		opts.MinReplicasLowerBound = 0
	}

	switch {
	case oldHPA != nil && oldHPA.Spec.ScaleTargetRef.APIVersion == newHPA.Spec.ScaleTargetRef.APIVersion && oldHPA.Spec.ScaleTargetRef.Kind == newHPA.Spec.ScaleTargetRef.Kind:
		// skip apiVersion validation on updates that don't change the kind/apiVersion.
		opts.ScaleTargetRefValidationOptions.AllowInvalidAPIVersion = true
	case newHPA.Spec.ScaleTargetRef.Kind == "ReplicationController":
		// allow empty apiVersion for the only scalable type that exists in the core v1 API.
		opts.ScaleTargetRefValidationOptions.AllowEmptyAPIGroup = true
	}

	if oldHPA != nil {
		for _, metric := range oldHPA.Spec.Metrics {
			if metric.Type == autoscaling.ObjectMetricSourceType && metric.Object != nil {
				if err := validation.ValidateAPIVersion(metric.Object.DescribedObject, opts.ObjectMetricsValidationOptions); err != nil {
					// metrics are already invalid.
					opts.ObjectMetricsValidationOptions.AllowInvalidAPIVersion = true
					break
				}
			}
		}
	}
	return opts
}

// dropDisabledFields will drop any disabled fields that have not previously been
// set on the old HPA. oldHPA is ignored if nil.
func dropDisabledFields(newHPA, oldHPA *autoscaling.HorizontalPodAutoscaler) {
	if utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableTolerance) {
		return
	}
	if toleranceInUse(oldHPA) {
		return
	}
	newBehavior := newHPA.Spec.Behavior
	if newBehavior == nil {
		return
	}

	for _, sr := range []*autoscaling.HPAScalingRules{newBehavior.ScaleDown, newBehavior.ScaleUp} {
		if sr != nil {
			sr.Tolerance = nil
		}
	}
}

func toleranceInUse(hpa *autoscaling.HorizontalPodAutoscaler) bool {
	if hpa == nil || hpa.Spec.Behavior == nil {
		return false
	}
	for _, sr := range []*autoscaling.HPAScalingRules{hpa.Spec.Behavior.ScaleDown, hpa.Spec.Behavior.ScaleUp} {
		if sr != nil && sr.Tolerance != nil {
			return true
		}
	}
	return false
}
