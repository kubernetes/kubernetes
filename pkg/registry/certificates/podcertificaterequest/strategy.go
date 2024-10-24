/*
Copyright 2024 The Kubernetes Authors.

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

// Package podcertificaterequest provides Registry interface and its RESTStorage
// implementation for storing PodCertificateRequest objects.
package podcertificaterequest // import "k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/certificates/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// strategy implements behavior for PodCertificateRequests.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the create, update, and delete strategy for PodCertificateRequests.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

var _ rest.RESTCreateStrategy = Strategy
var _ rest.RESTUpdateStrategy = Strategy
var _ rest.RESTDeleteStrategy = Strategy

func (strategy) NamespaceScoped() bool {
	return true
}

func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {}

func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	req := obj.(*certificates.PodCertificateRequest)
	return validation.ValidatePodCertificateRequest(req, validation.ValidatePodCertificateRequestOptions{})
}

func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (strategy) Canonicalize(obj runtime.Object) {}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (s strategy) PrepareForUpdate(ctx context.Context, new, old runtime.Object) {}

func (s strategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newReq := new.(*certificates.PodCertificateRequest)
	oldReq := old.(*certificates.PodCertificateRequest)
	return validation.ValidatePodCertificateRequestUpdate(newReq, oldReq, validation.ValidatePodCertificateRequestOptions{})
}

func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (strategy) AllowUnconditionalUpdate() bool {
	return false
}

// Storage stratehy for the status subresource
type statusStrategy struct {
	strategy
}

var StatusStrategy = statusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (statusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"certificates.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status", "conditions"),
		),
	}
	return fields
}

func (statusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newReq := obj.(*certificates.PodCertificateRequest)
	oldReq := old.(*certificates.PodCertificateRequest)

	// Updating /status should not modify spec
	newReq.Spec = oldReq.Spec

	// Specifically preserve existing Denied/Failed/SuggestedKeyType conditions.
	preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeDenied)
	preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeFailed)
	preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeSuggestedKeyType)

	populateConditionTimestamps(newReq, oldReq)
}

// preserveConditionInstances copies instances of the specified condition type from oldCSR to newCSR.
// or returns false if the newCSR added or removed instances
func preserveConditionInstances(newReq, oldReq *certificates.PodCertificateRequest, conditionType string) bool {
	oldIndices := findConditionIndices(oldReq, conditionType)
	newIndices := findConditionIndices(newReq, conditionType)
	if len(oldIndices) != len(newIndices) {
		// instances were added or removed, we cannot preserve the existing values
		return false
	}
	// preserve the old condition values
	for i, oldIndex := range oldIndices {
		newReq.Status.Conditions[newIndices[i]] = oldReq.Status.Conditions[oldIndex]
	}
	return true
}

// findConditionIndices returns the indices of instances of the specified condition type
func findConditionIndices(req *certificates.PodCertificateRequest, conditionType string) []int {
	var retval []int
	for i, c := range req.Status.Conditions {
		if c.Type == conditionType {
			retval = append(retval, i)
		}
	}
	return retval
}

// populateConditionTimestamps sets LastTransitionTime in newReq if missing
func populateConditionTimestamps(newReq, oldReq *certificates.PodCertificateRequest) {
	now := metav1.Now()
	for i := range newReq.Status.Conditions {

		// preserve existing lastTransitionTime if the condition with this type/status already exists,
		// otherwise set to now.
		if newReq.Status.Conditions[i].LastTransitionTime.IsZero() {
			lastTransition := now
			for _, oldCondition := range oldReq.Status.Conditions {
				if oldCondition.Type == newReq.Status.Conditions[i].Type &&
					oldCondition.Status == newReq.Status.Conditions[i].Status &&
					!oldCondition.LastTransitionTime.IsZero() {
					lastTransition = oldCondition.LastTransitionTime
					break
				}
			}
			newReq.Status.Conditions[i].LastTransitionTime = lastTransition
		}
	}
}

func (statusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodCertificateRequestStatusUpdate(obj.(*certificates.PodCertificateRequest), old.(*certificates.PodCertificateRequest))
}

// WarningsOnUpdate returns warnings for the given update.
func (statusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (statusStrategy) Canonicalize(obj runtime.Object) {}
