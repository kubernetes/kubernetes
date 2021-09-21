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
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/validation"
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
}

func (csiNodeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csiNode := obj.(*storage.CSINode)
	validateOptions := validation.CSINodeValidationOptions{
		AllowLongNodeID: true,
	}

	errs := validation.ValidateCSINode(csiNode, validateOptions)

	return errs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (csiNodeStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation.
func (csiNodeStrategy) Canonicalize(obj runtime.Object) {
}

func (csiNodeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate sets the driver's Allocatable fields that are not allowed to be set by an end user updating a CSINode.
func (csiNodeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (csiNodeStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCSINodeObj := obj.(*storage.CSINode)
	oldCSINodeObj := old.(*storage.CSINode)
	validateOptions := validation.CSINodeValidationOptions{
		AllowLongNodeID: true,
	}

	errorList := validation.ValidateCSINode(newCSINodeObj, validateOptions)
	return append(errorList, validation.ValidateCSINodeUpdate(newCSINodeObj, oldCSINodeObj, validateOptions)...)
}

// WarningsOnUpdate returns warnings for the given update.
func (csiNodeStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (csiNodeStrategy) AllowUnconditionalUpdate() bool {
	return false
}
