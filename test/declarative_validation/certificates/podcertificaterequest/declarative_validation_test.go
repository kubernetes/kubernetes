/*
Copyright The Kubernetes Authors.

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

package podcertificaterequest

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	registry "k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

type fakeAuthorizer struct{}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "default accept", nil
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:    "certificates.k8s.io",
				APIVersion:  apiVersion,
				Resource:    "podcertificaterequests",
				Subresource: "status",
			})
			ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test-user"})

			strategy := registry.NewStatusStrategy(registry.NewStrategy(), &fakeAuthorizer{}, nil)

			// Custom conditions for PCR to avoid "Unsupported value" from handwritten validation
			testCases := []meta.ConditionTestCase{
				{
					Name: "valid single condition",
					Conditions: []metav1.Condition{
						meta.MkCondition(meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied))),
					},
					ExpectedErrs: nil,
				},
				{
					Name: "invalid missing type",
					Conditions: []metav1.Condition{
						meta.MkCondition(meta.TweakType("")),
					},
					ExpectedErrs: field.ErrorList{
						field.Required(field.NewPath("status", "conditions").Index(0).Child("type"), "").MarkAlpha(),
						// handwritten validation doesn't support empty type and uses .[0] notation
						field.NotSupported(field.NewPath("status", "conditions").Child("[0]", "type"), "", []string{"Issued", "Denied", "Failed"}).MarkFromImperative(),
					},
				},
				{
					Name: "invalid duplicate types",
					Conditions: []metav1.Condition{
						meta.MkCondition(meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied))),
						meta.MkCondition(meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied))),
					},
					ExpectedErrs: field.ErrorList{
						field.Duplicate(field.NewPath("status", "conditions").Index(1), map[string]interface{}{"type": certificates.PodCertificateRequestConditionTypeDenied, "status": "True", "reason": "Foo", "message": "Bar"}).MarkAlpha(),
						// handwritten validation doesn't allow multiple known conditions and uses .[1] notation
						field.Invalid(field.NewPath("status", "conditions").Child("[1]", "type"), certificates.PodCertificateRequestConditionTypeDenied, `There may be at most one condition with type "Issued", "Denied", or "Failed"`).MarkFromImperative(),
					},
				},
				{
					Name: "invalid negative observedGeneration",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakObservedGeneration(-1),
						),
					},
					ExpectedErrs: field.ErrorList{
						field.Invalid(
							field.NewPath("status", "conditions").Index(0).Child("observedGeneration"),
							int64(-1),
							"",
						).WithOrigin("minimum").MarkAlpha(),
					},
				},
				{
					Name: "valid observedGeneration zero",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakObservedGeneration(0),
						),
					},
					ExpectedErrs: nil,
				},
			}

			for _, tc := range testCases {
				t.Run("conditions: "+tc.Name, func(t *testing.T) {
					obj := &certificates.PodCertificateRequest{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-pcr", Namespace: "default", ResourceVersion: "1"},
						Spec:       certificates.PodCertificateRequestSpec{},
						Status: certificates.PodCertificateRequestStatus{
							Conditions: tc.Conditions,
						},
					}
					old := &certificates.PodCertificateRequest{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-pcr", Namespace: "default", ResourceVersion: "1"},
						Spec:       certificates.PodCertificateRequestSpec{},
						Status:     certificates.PodCertificateRequestStatus{},
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, obj, old, strategy, tc.ExpectedErrs)
				})
			}
		})
	}
}
