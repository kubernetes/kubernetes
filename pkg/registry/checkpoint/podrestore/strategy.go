/*
Copyright 2026 The Kubernetes Authors.

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

package podrestore

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
	"k8s.io/kubernetes/pkg/apis/checkpoint/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// podRestoreStrategy implements verification logic for PodRestores.
type podRestoreStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodRestore objects.
var Strategy = podRestoreStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (podRestoreStrategy) NamespaceScoped() bool {
	return true
}

func (podRestoreStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"checkpoint.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

func (podRestoreStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pr := obj.(*checkpoint.PodRestore)
	pr.Status = checkpoint.PodRestoreStatus{
		Phase: checkpoint.PodRestorePending,
	}
	pr.Generation = 1
}

func (podRestoreStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPR := obj.(*checkpoint.PodRestore)
	oldPR := old.(*checkpoint.PodRestore)
	newPR.Status = oldPR.Status
}

func (podRestoreStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pr := obj.(*checkpoint.PodRestore)
	return validation.ValidatePodRestore(pr)
}

func (podRestoreStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (podRestoreStrategy) Canonicalize(obj runtime.Object) {
}

func (podRestoreStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (podRestoreStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodRestoreUpdate(obj.(*checkpoint.PodRestore), old.(*checkpoint.PodRestore))
}

func (podRestoreStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (podRestoreStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// podRestoreStatusStrategy implements verification logic for status updates of PodRestores.
type podRestoreStatusStrategy struct {
	podRestoreStrategy
}

// StatusStrategy is the logic for status updates of PodRestores.
var StatusStrategy = podRestoreStatusStrategy{Strategy}

func (podRestoreStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"checkpoint.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

func (podRestoreStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPR := obj.(*checkpoint.PodRestore)
	oldPR := old.(*checkpoint.PodRestore)
	newPR.Spec = oldPR.Spec
}

func (podRestoreStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}
