/*
Copyright 2019 The Kubernetes Authors.

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

package flowschema

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// flowSchemaStrategy implements verification logic for FlowSchema.
type flowSchemaStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating flow schema objects.
var Strategy = flowSchemaStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all PriorityClasses are global.
func (flowSchemaStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (flowSchemaStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"flowcontrol.apiserver.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"flowcontrol.apiserver.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"flowcontrol.apiserver.k8s.io/v1beta3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"flowcontrol.apiserver.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a flow-schema before creation.
func (flowSchemaStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	fl := obj.(*flowcontrol.FlowSchema)
	fl.Status = flowcontrol.FlowSchemaStatus{}
	fl.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (flowSchemaStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newFlowSchema := obj.(*flowcontrol.FlowSchema)
	oldFlowSchema := old.(*flowcontrol.FlowSchema)

	// Spec updates bump the generation so that we can distinguish between status updates.
	if !apiequality.Semantic.DeepEqual(newFlowSchema.Spec, oldFlowSchema.Spec) {
		newFlowSchema.Generation = oldFlowSchema.Generation + 1
	}
	newFlowSchema.Status = oldFlowSchema.Status
}

// Validate validates a new flow-schema.
func (flowSchemaStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateFlowSchema(obj.(*flowcontrol.FlowSchema))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (flowSchemaStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (flowSchemaStrategy) Canonicalize(obj runtime.Object) {
}

func (flowSchemaStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for flow-schemas; this means a POST is needed to create one.
func (flowSchemaStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (flowSchemaStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateFlowSchemaUpdate(old.(*flowcontrol.FlowSchema), obj.(*flowcontrol.FlowSchema))
}

// WarningsOnUpdate returns warnings for the given update.
func (flowSchemaStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type flowSchemaStatusStrategy struct {
	flowSchemaStrategy
}

// StatusStrategy is the default logic that applies when updating flow-schema objects' status.
var StatusStrategy = flowSchemaStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (flowSchemaStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"flowcontrol.apiserver.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"flowcontrol.apiserver.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"flowcontrol.apiserver.k8s.io/v1beta3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"flowcontrol.apiserver.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (flowSchemaStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newFlowSchema := obj.(*flowcontrol.FlowSchema)
	oldFlowSchema := old.(*flowcontrol.FlowSchema)

	// managedFields must be preserved since it's been modified to
	// track changed fields in the status update.
	managedFields := newFlowSchema.ManagedFields
	newFlowSchema.ObjectMeta = oldFlowSchema.ObjectMeta
	newFlowSchema.ManagedFields = managedFields
	newFlowSchema.Spec = oldFlowSchema.Spec
}

func (flowSchemaStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateFlowSchemaStatusUpdate(old.(*flowcontrol.FlowSchema), obj.(*flowcontrol.FlowSchema))
}

// WarningsOnUpdate returns warnings for the given update.
func (flowSchemaStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
