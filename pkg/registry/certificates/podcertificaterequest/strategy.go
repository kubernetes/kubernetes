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
	certadmission "k8s.io/kubernetes/plugin/pkg/admission/certificates"
	"k8s.io/utils/clock"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
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

	// TODO(KEP-4317): Drop the preserveConditionInstances,
	// populateConditionTimestamps.  Make sure this is covered by validation
	// unit tests.

	// Specifically preserve existing Denied/Failed/SuggestedKeyType conditions.
	// preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeIssued)
	// preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeDenied)
	// preserveConditionInstances(newReq, oldReq, certificates.PodCertificateRequestConditionTypeFailed)

	// populateConditionTimestamps(newReq, oldReq)
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

func (s *StatusStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	oldPCR := old.(*certificates.PodCertificateRequest)
	newPCR := new.(*certificates.PodCertificateRequest)

	errs := certvalidation.ValidatePodCertificateRequestStatusUpdate(newPCR, oldPCR, s.clock)
	if len(errs) != 0 {
		return errs
	}

	// If the caller is trying to change status.CertificateChain, they must have
	// the appropriate "sign" permission on the requested signername.
	if oldPCR.Status.CertificateChain != newPCR.Status.CertificateChain {
		user, ok := genericapirequest.UserFrom(ctx)
		if !ok {
			return field.ErrorList{
				field.InternalError(field.NewPath("spec", "signerName"), fmt.Errorf("cannot determine calling user to perform \"sign\" check")),
			}
		}

		if !certadmission.IsAuthorizedForSignerName(ctx, s.authorizer, user, "sign", oldPCR.Spec.SignerName) {
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
