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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
)

// csiNodeStrategy implements behavior for CSINode objects
type csiNodeStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// CSINode objects via the REST API.
var Strategy = csiNodeStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (csiNodeStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears fields that are not allowed to be set on creation.
func (csiNodeStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	csiNode := obj.(*storage.CSINode)
	if !utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		for i := range csiNode.Spec.Drivers {
			csiNode.Spec.Drivers[i].Allocatable = nil
		}
	}
}

func (csiNodeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csiNode := obj.(*storage.CSINode)

	errs := validation.ValidateCSINode(csiNode)
	errs = append(errs, validation.ValidateCSINode(csiNode)...)

	return errs
}

// Canonicalize normalizes the object after validation.
func (csiNodeStrategy) Canonicalize(obj runtime.Object) {
}

func (csiNodeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the driver's Allocatable fields that are not allowed to be set by an end user updating a CSINode.
func (csiNodeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSINode := obj.(*storage.CSINode)
	oldCSINode := old.(*storage.CSINode)

	inUse := getAllocatablesInUse(oldCSINode)

	if !utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		for i := range newCSINode.Spec.Drivers {
			if !inUse[newCSINode.Spec.Drivers[i].Name] {
				newCSINode.Spec.Drivers[i].Allocatable = nil
			}
		}
	}
}

func getAllocatablesInUse(obj *storage.CSINode) map[string]bool {
	inUse := make(map[string]bool)
	if obj == nil {
		return inUse
	}
	for i := range obj.Spec.Drivers {
		if obj.Spec.Drivers[i].Allocatable != nil {
			inUse[obj.Spec.Drivers[i].Name] = true
		}
	}
	return inUse
}

func (csiNodeStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSINodeObj := obj.(*storage.CSINode)
	oldCSINodeObj := old.(*storage.CSINode)
	errorList := validation.ValidateCSINode(newCSINodeObj)
	return append(errorList, validation.ValidateCSINodeUpdate(newCSINodeObj, oldCSINodeObj)...)
}

func (csiNodeStrategy) AllowUnconditionalUpdate() bool {
	return false
}
