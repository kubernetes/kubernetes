/*
Copyright 2018 The Kubernetes Authors.

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

package podgroup

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/scheduling/validation"
)

// podGroupStrategy implements verification logic for PodGroup.
type podGroupStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodGroup objects.
var Strategy = podGroupStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because PodGroup is namespaced type.
func (podGroupStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a PodGroup before creation.
func (podGroupStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pc := obj.(*scheduling.PodGroup)
	pc.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podGroupStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	_ = obj.(*scheduling.PodGroup)
	_ = old.(*scheduling.PodGroup)
}

// Validate validates a new PodGroup.
func (podGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pg := obj.(*scheduling.PodGroup)
	return validation.ValidatePodGroup(pg)
}

// Canonicalize normalizes the object after validation.
func (podGroupStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for PodGroup; this means POST is needed to create one.
func (podGroupStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodGroupUpdate(obj.(*scheduling.PodGroup), old.(*scheduling.PodGroup))
}

// AllowUnconditionalUpdate is the default update policy for PodGroup objects.
func (podGroupStrategy) AllowUnconditionalUpdate() bool {
	return true
}
