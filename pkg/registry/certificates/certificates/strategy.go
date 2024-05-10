/*
Copyright 2016 The Kubernetes Authors.

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

package certificates

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/certificates/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// csrStrategy implements behavior for CSRs
type csrStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// csrStrategy is the default logic that applies when creating and updating
// CSR objects.
var Strategy = csrStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is false for CSRs.
func (csrStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (csrStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"certificates.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status"),
		),
		"certificates.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// AllowCreateOnUpdate is false for CSRs.
func (csrStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (csrStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	csr := obj.(*certificates.CertificateSigningRequest)

	// Clear any user-specified info
	csr.Spec.Username = ""
	csr.Spec.UID = ""
	csr.Spec.Groups = nil
	csr.Spec.Extra = nil
	// Inject user.Info from request context
	if user, ok := genericapirequest.UserFrom(ctx); ok {
		csr.Spec.Username = user.GetName()
		csr.Spec.UID = user.GetUID()
		csr.Spec.Groups = user.GetGroups()
		if extra := user.GetExtra(); len(extra) > 0 {
			csr.Spec.Extra = map[string]certificates.ExtraValue{}
			for k, v := range extra {
				csr.Spec.Extra[k] = v
			}
		}
	}

	// Be explicit that users cannot create pre-approved certificate requests.
	csr.Status = certificates.CertificateSigningRequestStatus{}
	csr.Status.Conditions = []certificates.CertificateSigningRequestCondition{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users
// on update. Certificate requests are immutable after creation except via subresources.
func (csrStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	newCSR.Spec = oldCSR.Spec
	newCSR.Status = oldCSR.Status
}

// Validate validates a new CSR. Validation must check for a correct signature.
func (csrStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	csr := obj.(*certificates.CertificateSigningRequest)
	return validation.ValidateCertificateSigningRequestCreate(csr)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (csrStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation (which includes a signature check).
func (csrStrategy) Canonicalize(obj runtime.Object) {}

// ValidateUpdate is the default update validation for an end user.
func (csrStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	oldCSR := old.(*certificates.CertificateSigningRequest)
	newCSR := obj.(*certificates.CertificateSigningRequest)
	return validation.ValidateCertificateSigningRequestUpdate(newCSR, oldCSR)
}

// WarningsOnUpdate returns warnings for the given update.
func (csrStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// If AllowUnconditionalUpdate() is true and the object specified by
// the user does not have a resource version, then generic Update()
// populates it with the latest version. Else, it checks that the
// version specified by the user matches the version of latest etcd
// object.
func (csrStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// Storage strategy for the Status subresource
type csrStatusStrategy struct {
	csrStrategy
}

var StatusStrategy = csrStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (csrStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"certificates.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status", "conditions"),
		),
		"certificates.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status", "conditions"),
		),
	}

	return fields
}

func (csrStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	// Updating /status should not modify spec
	newCSR.Spec = oldCSR.Spec

	// Specifically preserve existing Approved/Denied conditions.
	// Adding/removing Approved/Denied conditions will cause these to fail,
	// and the change in Approved/Denied conditions will produce a validation error
	preserveConditionInstances(newCSR, oldCSR, certificates.CertificateApproved)
	preserveConditionInstances(newCSR, oldCSR, certificates.CertificateDenied)

	populateConditionTimestamps(newCSR, oldCSR)
}

// preserveConditionInstances copies instances of the specified condition type from oldCSR to newCSR.
// or returns false if the newCSR added or removed instances
func preserveConditionInstances(newCSR, oldCSR *certificates.CertificateSigningRequest, conditionType certificates.RequestConditionType) bool {
	oldIndices := findConditionIndices(oldCSR, conditionType)
	newIndices := findConditionIndices(newCSR, conditionType)
	if len(oldIndices) != len(newIndices) {
		// instances were added or removed, we cannot preserve the existing values
		return false
	}
	// preserve the old condition values
	for i, oldIndex := range oldIndices {
		newCSR.Status.Conditions[newIndices[i]] = oldCSR.Status.Conditions[oldIndex]
	}
	return true
}

// findConditionIndices returns the indices of instances of the specified condition type
func findConditionIndices(csr *certificates.CertificateSigningRequest, conditionType certificates.RequestConditionType) []int {
	var retval []int
	for i, c := range csr.Status.Conditions {
		if c.Type == conditionType {
			retval = append(retval, i)
		}
	}
	return retval
}

// nowFunc allows overriding for unit tests
var nowFunc = metav1.Now

// populateConditionTimestamps sets LastUpdateTime and LastTransitionTime in newCSR if missing
func populateConditionTimestamps(newCSR, oldCSR *certificates.CertificateSigningRequest) {
	now := nowFunc()
	for i := range newCSR.Status.Conditions {
		if newCSR.Status.Conditions[i].LastUpdateTime.IsZero() {
			newCSR.Status.Conditions[i].LastUpdateTime = now
		}

		// preserve existing lastTransitionTime if the condition with this type/status already exists,
		// otherwise set to now.
		if newCSR.Status.Conditions[i].LastTransitionTime.IsZero() {
			lastTransition := now
			for _, oldCondition := range oldCSR.Status.Conditions {
				if oldCondition.Type == newCSR.Status.Conditions[i].Type &&
					oldCondition.Status == newCSR.Status.Conditions[i].Status &&
					!oldCondition.LastTransitionTime.IsZero() {
					lastTransition = oldCondition.LastTransitionTime
					break
				}
			}
			newCSR.Status.Conditions[i].LastTransitionTime = lastTransition
		}
	}
}

func (csrStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCertificateSigningRequestStatusUpdate(obj.(*certificates.CertificateSigningRequest), old.(*certificates.CertificateSigningRequest))
}

// WarningsOnUpdate returns warnings for the given update.
func (csrStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (csrStatusStrategy) Canonicalize(obj runtime.Object) {
}

// Storage strategy for the Approval subresource
type csrApprovalStrategy struct {
	csrStrategy
}

var ApprovalStrategy = csrApprovalStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (csrApprovalStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"certificates.k8s.io/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status", "certificate"),
		),
		"certificates.k8s.io/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("status", "certificate"),
		),
	}

	return fields
}

// PrepareForUpdate prepares the new certificate signing request by limiting
// the data that is updated to only the conditions and populating condition timestamps
func (csrApprovalStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	populateConditionTimestamps(newCSR, oldCSR)
	newConditions := newCSR.Status.Conditions

	// Updating the approval should only update the conditions.
	newCSR.Spec = oldCSR.Spec
	newCSR.Status = oldCSR.Status
	newCSR.Status.Conditions = newConditions
}

func (csrApprovalStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCertificateSigningRequestApprovalUpdate(obj.(*certificates.CertificateSigningRequest), old.(*certificates.CertificateSigningRequest))
}

// WarningsOnUpdate returns warnings for the given update.
func (csrApprovalStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
