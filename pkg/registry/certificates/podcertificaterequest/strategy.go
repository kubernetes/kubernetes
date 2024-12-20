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
	"fmt"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certvalidation "k8s.io/kubernetes/pkg/apis/certificates/validation"
	"k8s.io/kubernetes/pkg/certauthorization"
	"k8s.io/utils/clock"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// strategy implements behavior for PodCertificateRequests.
type Strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

var _ rest.RESTCreateStrategy = (*Strategy)(nil)
var _ rest.RESTUpdateStrategy = (*Strategy)(nil)
var _ rest.RESTDeleteStrategy = (*Strategy)(nil)

func NewStrategy() *Strategy {
	return &Strategy{
		ObjectTyper:   legacyscheme.Scheme,
		NameGenerator: names.SimpleNameGenerator,
	}
}

func (s *Strategy) NamespaceScoped() bool {
	return true
}

func (s *Strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	req := obj.(*certificates.PodCertificateRequest)
	req.Status = certificates.PodCertificateRequestStatus{}
}

func (s *Strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	req := obj.(*certificates.PodCertificateRequest)
	return certvalidation.ValidatePodCertificateRequestCreate(req)
}

func (s *Strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (s *Strategy) Canonicalize(obj runtime.Object) {}

func (s *Strategy) AllowCreateOnUpdate() bool {
	return false
}

func (s *Strategy) PrepareForUpdate(ctx context.Context, new, old runtime.Object) {
	newReq := new.(*certificates.PodCertificateRequest)
	oldReq := old.(*certificates.PodCertificateRequest)
	newReq.Status = oldReq.Status
}

func (s *Strategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newReq := new.(*certificates.PodCertificateRequest)
	oldReq := old.(*certificates.PodCertificateRequest)
	return certvalidation.ValidatePodCertificateRequestUpdate(newReq, oldReq)
}

func (s *Strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (s *Strategy) AllowUnconditionalUpdate() bool {
	return false
}

// StatusStrategy is the strategy for the status subresource.
type StatusStrategy struct {
	*Strategy
	authorizer authorizer.Authorizer
	clock      clock.PassiveClock
}

func NewStatusStrategy(strategy *Strategy, authorizer authorizer.Authorizer, clock clock.PassiveClock) *StatusStrategy {
	return &StatusStrategy{
		Strategy:   strategy,
		authorizer: authorizer,
		clock:      clock,
	}
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (s *StatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"certificates.k8s.io/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

func (s *StatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newReq := obj.(*certificates.PodCertificateRequest)
	oldReq := old.(*certificates.PodCertificateRequest)

	// Updating /status should not modify spec
	newReq.Spec = oldReq.Spec

	metav1.ResetObjectMetaForStatus(&newReq.ObjectMeta, &oldReq.ObjectMeta)
}

func (s *StatusStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	oldPCR := old.(*certificates.PodCertificateRequest)
	newPCR := new.(*certificates.PodCertificateRequest)

	errs := certvalidation.ValidatePodCertificateRequestStatusUpdate(newPCR, oldPCR, s.clock)
	if len(errs) != 0 {
		return errs
	}

	// If the caller is trying to change any status fields, they must have
	// the appropriate "sign" permission on the requested signername.
	if !equality.Semantic.DeepEqual(oldPCR.Status, newPCR.Status) {
		user, ok := genericapirequest.UserFrom(ctx)
		if !ok {
			return field.ErrorList{
				field.InternalError(field.NewPath("spec", "signerName"), fmt.Errorf("cannot determine calling user to perform \"sign\" check")),
			}
		}

		if !certauthorization.IsAuthorizedForSignerName(ctx, s.authorizer, user, "sign", oldPCR.Spec.SignerName) {
			klog.V(4).Infof("user not permitted to sign PodCertificateRequest %q with signerName %q", oldPCR.Name, oldPCR.Spec.SignerName)
			return field.ErrorList{
				field.Forbidden(field.NewPath("spec", "signerName"), fmt.Sprintf("User %q is not permitted to \"sign\" for signer %q", user.GetName(), oldPCR.Spec.SignerName)),
			}
		}
	}

	return nil
}

// WarningsOnUpdate returns warnings for the given update.
func (s *StatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (s *StatusStrategy) Canonicalize(obj runtime.Object) {}
