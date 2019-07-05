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

package priorityclass

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingutil "k8s.io/kubernetes/pkg/apis/scheduling/util"
	"k8s.io/kubernetes/pkg/apis/scheduling/validation"
)

// priorityClassStrategy implements verification logic for PriorityClass.
type priorityClassStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PriorityClass objects.
var Strategy = priorityClassStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all PriorityClasses are global.
func (priorityClassStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of a PriorityClass before creation.
func (priorityClassStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pc := obj.(*scheduling.PriorityClass)
	pc.Generation = 1
	schedulingutil.DropDisabledFields(pc, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (priorityClassStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClass := obj.(*scheduling.PriorityClass)
	oldClass := old.(*scheduling.PriorityClass)

	schedulingutil.DropDisabledFields(newClass, oldClass)
}

// Validate validates a new PriorityClass.
func (priorityClassStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pc := obj.(*scheduling.PriorityClass)
	return validation.ValidatePriorityClass(pc)
}

// Canonicalize normalizes the object after validation.
func (priorityClassStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for PriorityClass; this means POST is needed to create one.
func (priorityClassStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (priorityClassStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePriorityClassUpdate(obj.(*scheduling.PriorityClass), old.(*scheduling.PriorityClass))
}

// AllowUnconditionalUpdate is the default update policy for PriorityClass objects.
func (priorityClassStrategy) AllowUnconditionalUpdate() bool {
	return true
}
