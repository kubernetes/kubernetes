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
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/certificates/validation"
)

// csrStrategy implements behavior for CSRs
type csrStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// csrStrategy is the default logic that applies when creating and updating
// CSR objects.
var Strategy = csrStrategy{api.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for CSRs.
func (csrStrategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate is false for CSRs.
func (csrStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForCreate clears fields that are not allowed to be set by end users
// on creation.
func (csrStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	csr := obj.(*certificates.CertificateSigningRequest)

	// Clear any user-specified info
	csr.Spec.Username = ""
	csr.Spec.UID = ""
	csr.Spec.Groups = nil
	// Inject user.Info from request context
	if user, ok := genericapirequest.UserFrom(ctx); ok {
		csr.Spec.Username = user.GetName()
		csr.Spec.UID = user.GetUID()
		csr.Spec.Groups = user.GetGroups()
	}

	// Be explicit that users cannot create pre-approved certificate requests.
	csr.Status = certificates.CertificateSigningRequestStatus{}
	csr.Status.Conditions = []certificates.CertificateSigningRequestCondition{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users
// on update. Certificate requests are immutable after creation except via subresources.
func (csrStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	newCSR.Spec = oldCSR.Spec
	newCSR.Status = oldCSR.Status
}

// Validate validates a new CSR. Validation must check for a correct signature.
func (csrStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	csr := obj.(*certificates.CertificateSigningRequest)
	return validation.ValidateCertificateSigningRequest(csr)
}

// Canonicalize normalizes the object after validation (which includes a signature check).
func (csrStrategy) Canonicalize(obj runtime.Object) {}

// ValidateUpdate is the default update validation for an end user.
func (csrStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	oldCSR := old.(*certificates.CertificateSigningRequest)
	newCSR := obj.(*certificates.CertificateSigningRequest)
	return validation.ValidateCertificateSigningRequestUpdate(newCSR, oldCSR)
}

// If AllowUnconditionalUpdate() is true and the object specified by
// the user does not have a resource version, then generic Update()
// populates it with the latest version. Else, it checks that the
// version specified by the user matches the version of latest etcd
// object.
func (csrStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func (s csrStrategy) Export(ctx genericapirequest.Context, obj runtime.Object, exact bool) error {
	csr, ok := obj.(*certificates.CertificateSigningRequest)
	if !ok {
		// unexpected programmer error
		return fmt.Errorf("unexpected object: %v", obj)
	}
	s.PrepareForCreate(ctx, obj)
	if exact {
		return nil
	}
	// CSRs allow direct subresource edits, we clear them without exact so the CSR value can be reused.
	csr.Status = certificates.CertificateSigningRequestStatus{}
	return nil
}

// Storage strategy for the Status subresource
type csrStatusStrategy struct {
	csrStrategy
}

var StatusStrategy = csrStatusStrategy{Strategy}

func (csrStatusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	// Updating the Status should only update the Status and not the spec
	// or approval conditions. The intent is to separate the concerns of
	// approval and certificate issuance.
	newCSR.Spec = oldCSR.Spec
	newCSR.Status.Conditions = oldCSR.Status.Conditions
}

func (csrStatusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCertificateSigningRequestUpdate(obj.(*certificates.CertificateSigningRequest), old.(*certificates.CertificateSigningRequest))
}

// Canonicalize normalizes the object after validation.
func (csrStatusStrategy) Canonicalize(obj runtime.Object) {
}

// Storage strategy for the Approval subresource
type csrApprovalStrategy struct {
	csrStrategy
}

var ApprovalStrategy = csrApprovalStrategy{Strategy}

func (csrApprovalStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newCSR := obj.(*certificates.CertificateSigningRequest)
	oldCSR := old.(*certificates.CertificateSigningRequest)

	// Updating the approval should only update the conditions.
	newCSR.Spec = oldCSR.Spec
	oldCSR.Status.Conditions = newCSR.Status.Conditions
	newCSR.Status = oldCSR.Status
}

func (csrApprovalStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCertificateSigningRequestUpdate(obj.(*certificates.CertificateSigningRequest), old.(*certificates.CertificateSigningRequest))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	sa, ok := obj.(*certificates.CertificateSigningRequest)
	if !ok {
		return nil, nil, fmt.Errorf("not a CertificateSigningRequest")
	}
	return labels.Set(sa.Labels), SelectableFields(sa), nil
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// SelectableFields returns a field set that can be used for filter selection
func SelectableFields(obj *certificates.CertificateSigningRequest) fields.Set {
	return generic.ObjectMetaFieldsSet(&obj.ObjectMeta, false)
}
