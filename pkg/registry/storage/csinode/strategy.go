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

package csinode

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// csiNodeStrategy implements behavior for CSINode objects
type csiNodeStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// CSINode objects via the REST API.
var Strategy = csiNodeStrategy{rest.DeclarativeValidation{Scheme: legacyscheme.Scheme}, names.SimpleNameGenerator}

func (csiNodeStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (csiNodeStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storage.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears fields that are not allowed to be set on creation.
func (csiNodeStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	csiNode := obj.(*storage.CSINode)
	csiNode.Status = storage.CSINodeStatus{}
	dropDisabledCSINodeFields(csiNode, nil)
}

func (csiNodeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csiNode := obj.(*storage.CSINode)
	return validation.ValidateCSINode(csiNode)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (csiNodeStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation.
func (csiNodeStrategy) Canonicalize(obj runtime.Object) {
}

func (csiNodeStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

// PrepareForUpdate sets the driver's Allocatable fields that are not allowed to be set by an end user updating a CSINode.
func (csiNodeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSINode := obj.(*storage.CSINode)
	oldCSINode := old.(*storage.CSINode)
	newCSINode.Status = oldCSINode.Status
	dropDisabledCSINodeFields(newCSINode, oldCSINode)
}

func (csiNodeStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSINodeObj := obj.(*storage.CSINode)
	oldCSINodeObj := old.(*storage.CSINode)
	return validation.ValidateCSINodeUpdate(newCSINodeObj, oldCSINodeObj)
}

// WarningsOnUpdate returns warnings for the given update.
func (csiNodeStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (csiNodeStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}

// csiNodeStatusStrategy implements behavior for CSINode status subresource
type csiNodeStatusStrategy struct {
	csiNodeStrategy
}

// StatusStrategy is the default logic that applies when updating
// CSINode status subresource via the REST API.
var StatusStrategy = csiNodeStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (csiNodeStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"storage.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

// PrepareForUpdate preserves spec and resets metadata for status updates.
func (csiNodeStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSINode := obj.(*storage.CSINode)
	oldCSINode := old.(*storage.CSINode)
	newCSINode.Spec = oldCSINode.Spec
	metav1.ResetObjectMetaForStatus(&newCSINode.ObjectMeta, &oldCSINode.ObjectMeta)
	dropDisabledCSINodeFields(newCSINode, oldCSINode)
}

func (csiNodeStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSINode := obj.(*storage.CSINode)
	oldCSINode := old.(*storage.CSINode)
	return validation.ValidateCSINodeStatusUpdate(newCSINode, oldCSINode)
}

func dropDisabledCSINodeFields(newObj, oldObj *storage.CSINode) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIVolumeHealth) {
		if oldObj == nil || len(oldObj.Status.StorageHealth) == 0 {
			newObj.Status.StorageHealth = nil
		}
	}
}
