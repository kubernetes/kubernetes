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
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	certificates "k8s.io/kubernetes/pkg/apis/certificates"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	registry "k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

type fakeAuthorizer struct{}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "default accept", nil
}

// Helper function to create a baseline valid PodCertificateRequest with optional tweaks
func mkPodCertificateRequest(tweaks ...func(*certificates.PodCertificateRequest)) certificates.PodCertificateRequest {
	obj := func() certificates.PodCertificateRequest {
		expiration := int32(3600)
		_, priv, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			panic(err)
		}
		template := &x509.CertificateRequest{}
		csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, priv)
		if err != nil {
			panic(err)
		}
		return certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Name: "valid-resource-name",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "kubernetes.io/kube-apiserver-client-pod",
				PodName:              "valid-pod-name",
				PodUID:               types.UID("a0123456-7890-abcd-ef01-234567890abc"),
				ServiceAccountName:   "default",
				ServiceAccountUID:    types.UID("b0123456-7890-abcd-ef01-234567890abc"),
				NodeName:             types.NodeName("valid-node-name"),
				NodeUID:              types.UID("c0123456-7890-abcd-ef01-234567890abc"),
				MaxExpirationSeconds: &expiration,
				StubPKCS10Request:    csrDER,
			},
		}
	}()
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy()
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "podcertificaterequests",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkPodCertificateRequest(func(o *certificates.PodCertificateRequest) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy()
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "podcertificaterequests",
				Namespace:         namespace,
				Name:              "valid-resource-name",
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkPodCertificateRequest(func(o *certificates.PodCertificateRequest) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "podcertificaterequests",
				Subresource:       "status",
				Name:              "valid-resource-name",
				IsResourceRequest: true,
				Verb:              "update",
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
				{
					Name: "invalid missing status",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakStatus(""),
						),
					},
					ExpectedErrs: field.ErrorList{
						field.Required(field.NewPath("status", "conditions").Index(0).Child("status"), "").MarkAlpha(),
						// handwritten validation requires "True"
						field.NotSupported(field.NewPath("status", "conditions").Child("[0]", "status"), "", []string{"True"}).MarkFromImperative(),
					},
				},
				{
					Name: "invalid status value",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakStatus("Invalid"),
						),
					},
					ExpectedErrs: field.ErrorList{
						field.NotSupported(
							field.NewPath("status", "conditions").Index(0).Child("status"),
							metav1.ConditionStatus("Invalid"),
							[]string{"False", "True", "Unknown"},
						).MarkAlpha(),
						// handwritten validation requires "True"
						field.NotSupported(field.NewPath("status", "conditions").Child("[0]", "status"), "Invalid", []string{"True"}).MarkFromImperative(),
					},
				},
				{
					Name: "invalid status value: Unknown",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakStatus("Unknown"),
						),
					},
					ExpectedErrs: field.ErrorList{
						// handwritten validation requires "True"
						field.NotSupported(field.NewPath("status", "conditions").Child("[0]", "status"), "Unknown", []string{"True"}).MarkFromImperative(),
					},
				},
				{
					Name: "invalid status value: False",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType(string(certificates.PodCertificateRequestConditionTypeDenied)),
							meta.TweakStatus("False"),
						),
					},
					ExpectedErrs: field.ErrorList{
						// handwritten validation requires "True"
						field.NotSupported(field.NewPath("status", "conditions").Child("[0]", "status"), "False", []string{"True"}).MarkFromImperative(),
					},
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
