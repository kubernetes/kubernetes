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

package resourcepoolstatusrequest

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// resourcePoolStatusRequestStrategy implements behavior for ResourcePoolStatusRequest objects
type resourcePoolStatusRequestStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var (
	Strategy       = &resourcePoolStatusRequestStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}
	StatusStrategy = &resourcePoolStatusRequestStatusStrategy{resourcePoolStatusRequestStrategy: Strategy}
)

func (resourcePoolStatusRequestStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new ResourcePoolStatusRequest that is the
// status.
func (*resourcePoolStatusRequestStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*resourcePoolStatusRequestStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	request := obj.(*resource.ResourcePoolStatusRequest)
	// Status must not be set by user on create.
	request.Status = resource.ResourcePoolStatusRequestStatus{}
}

func (*resourcePoolStatusRequestStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	request := obj.(*resource.ResourcePoolStatusRequest)
	return validation.ValidateResourcePoolStatusRequest(request)
}

func (*resourcePoolStatusRequestStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*resourcePoolStatusRequestStrategy) Canonicalize(obj runtime.Object) {
}

func (*resourcePoolStatusRequestStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*resourcePoolStatusRequestStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	request := obj.(*resource.ResourcePoolStatusRequest)
	oldRequest := old.(*resource.ResourcePoolStatusRequest)
	// Spec is immutable after creation
	request.Spec = oldRequest.Spec
	// Status is not updated via the main resource endpoint
	request.Status = oldRequest.Status
}

func (*resourcePoolStatusRequestStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateResourcePoolStatusRequestUpdate(obj.(*resource.ResourcePoolStatusRequest), old.(*resource.ResourcePoolStatusRequest))
}

func (*resourcePoolStatusRequestStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*resourcePoolStatusRequestStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type resourcePoolStatusRequestStatusStrategy struct {
	*resourcePoolStatusRequestStrategy
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*resourcePoolStatusRequestStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*resourcePoolStatusRequestStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRequest := obj.(*resource.ResourcePoolStatusRequest)
	oldRequest := old.(*resource.ResourcePoolStatusRequest)
	newRequest.Spec = oldRequest.Spec
	metav1.ResetObjectMetaForStatus(&newRequest.ObjectMeta, &oldRequest.ObjectMeta)
}

func (r *resourcePoolStatusRequestStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newRequest := obj.(*resource.ResourcePoolStatusRequest)
	oldRequest := old.(*resource.ResourcePoolStatusRequest)
	return validation.ValidateResourcePoolStatusRequestStatusUpdate(newRequest, oldRequest)
}

// WarningsOnUpdate returns warnings for the given update.
func (*resourcePoolStatusRequestStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
