/*
Copyright 2025 The Kubernetes Authors.

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
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

var apiVersions = []string{"v1", "v1beta1"}

type validationStrategy interface {
	Validate(ctx context.Context, obj runtime.Object) field.ErrorList
	ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList
}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "certificates.k8s.io",
		APIVersion: apiVersion,
	})
	testCases := map[string]struct {
		input        api.CertificateSigningRequest
		expectedErrs field.ErrorList
	}{
		"status.conditions: none = valid": {
			input: makeValidCSR(),
		},
		"status.conditions: Approved = valid": {
			input: makeValidCSR(withApprovedCondition()),
		},
		"status.conditions: Denied = valid": {
			input: makeValidCSR(withDeniedCondition()),
		},
		"status.conditions: Failed = valid": {
			input: makeValidCSR(withFailedCondition()),
		},
		"status.conditions: Approved+Failed = valid": {
			input: makeValidCSR(withApprovedCondition(), withFailedCondition()),
		},
		"status.conditions: Denied+Failed = valid": {
			input: makeValidCSR(withDeniedCondition(), withFailedCondition()),
		},
		"status.conditions: Approved+Denied = invalid": {
			input: makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions"), nil, "").WithOrigin("zeroOrOneOf"),
			},
		},
		"status.conditions: Denied+Approved = invalid": {
			input: makeValidCSR(withDeniedCondition(), withApprovedCondition()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions"), nil, "").WithOrigin("zeroOrOneOf"),
			},
		},
		"spec.usages: nil = invalid": {
			input: makeValidCSR(func(csr *api.CertificateSigningRequest) {
				csr.Spec.Usages = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "usages"), ""),
			},
		},
		"spec.usages: empty = invalid": {
			input: makeValidCSR(func(csr *api.CertificateSigningRequest) {
				csr.Spec.Usages = []api.KeyUsage{}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "usages"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		old          api.CertificateSigningRequest
		update       api.CertificateSigningRequest
		expectedErrs field.ErrorList
		subresources []string
	}{
		"no change in conditions - valid": {
			old:          makeValidCSR(withApprovedCondition()),
			update:       makeValidCSR(withApprovedCondition()),
			subresources: []string{"/", "/approval", "/status"},
		},
		"ratcheting: approved+denied conditions unchanged - valid": {
			old:          makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			update:       makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			subresources: []string{"/", "/approval", "/status"},
		},
		"ratcheting: approved+denied conditions, change spec - valid": {
			old: makeValidCSR(
				withApprovedCondition(),
				withDeniedCondition(),
				func(csr *api.CertificateSigningRequest) {
					csr.Spec.ExpirationSeconds = ptr.To(int32(3600))
				},
			),
			update: makeValidCSR(
				withApprovedCondition(),
				withDeniedCondition(),
				func(csr *api.CertificateSigningRequest) {
					csr.Spec.ExpirationSeconds = ptr.To(int32(7200))
				},
			),
			subresources: []string{"/", "/approval", "/status"},
		},
		"ratcheting: approved+denied conditions, add failed condition - valid": {
			old:          makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			update:       makeValidCSR(withApprovedCondition(), withDeniedCondition(), withFailedCondition()),
			subresources: []string{"/", "/approval", "/status"},
		},
		"ratcheting: approved+denied conditions, swapped order - valid": {
			old:          makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			update:       makeValidCSR(withDeniedCondition(), withApprovedCondition()),
			subresources: []string{"/", "/approval", "/status"},
		},
		"add approved condition - valid": {
			old:          makeValidCSR(),
			update:       makeValidCSR(withApprovedCondition()),
			subresources: []string{"/approval"}, // Can only add Approved and Denied conditions on /approval subresource
		},
		"add approved+denied conditions - invalid": {
			old:    makeValidCSR(),
			update: makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions"), nil, "").WithOrigin("zeroOrOneOf"),
			},
			subresources: []string{"/approval"}, // Can only add Approved and Denied conditions on /approval subresource
		},
		"ratcheting: approved+denied conditions, modify condition reason - valid": {
			old: makeValidCSR(
				func(csr *api.CertificateSigningRequest) {
					csr.Status.Conditions = []api.CertificateSigningRequestCondition{
						{Type: api.CertificateApproved, Status: core.ConditionTrue, Reason: "OldReason"},
						{Type: api.CertificateDenied, Status: core.ConditionTrue, Reason: "OldReason"},
					}
				},
			),
			update: makeValidCSR(
				func(csr *api.CertificateSigningRequest) {
					csr.Status.Conditions = []api.CertificateSigningRequestCondition{
						{Type: api.CertificateApproved, Status: core.ConditionTrue, Reason: "NewReason"},
						{Type: api.CertificateDenied, Status: core.ConditionTrue, Reason: "NewReason"},
					}
				},
			),
			subresources: []string{"/approval"}, // Can only modify Approved and Denied conditions on /approval subresource
		},
		"ratcheting: allow existing duplicate types - valid": {
			old:          makeValidCSR(withApprovedCondition(), withApprovedCondition(), withDeniedCondition(), withDeniedCondition()),
			update:       makeValidCSR(withDeniedCondition(), withDeniedCondition(), withApprovedCondition(), withApprovedCondition()),
			subresources: []string{"/status"},
		},
	}

	for k, tc := range testCases {
		for _, subresource := range tc.subresources {
			t.Run(k+" subresource="+subresource, func(t *testing.T) {
				ctx := createContextForSubresource(apiVersion, subresource)
				var strategy validationStrategy
				switch subresource {
				case "/":
					strategy = Strategy
				case "/approval":
					strategy = ApprovalStrategy
				case "/status":
					strategy = StatusStrategy
				}

				tc.old.ResourceVersion = "1"
				tc.update.ResourceVersion = "1"
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy.ValidateUpdate, tc.expectedErrs)
			})
		}
	}
}

func createContextForSubresource(apiVersion, subresource string) context.Context {
	requestInfo := &genericapirequest.RequestInfo{
		APIGroup:   "certificates.k8s.io",
		APIVersion: apiVersion,
	}

	if subresource != "/" {
		requestInfo.Subresource = subresource[1:] // Remove leading "/"
	}

	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), requestInfo)
}

func makeValidCSR(mutators ...func(*api.CertificateSigningRequest)) api.CertificateSigningRequest {
	csr := api.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-csr",
		},
		Spec: api.CertificateSigningRequestSpec{
			Request:    newCSRPEM(&testing.T{}),
			SignerName: "example.com/signer",
			Usages:     []api.KeyUsage{api.UsageDigitalSignature, api.UsageKeyEncipherment},
		},
	}
	for _, mutate := range mutators {
		mutate(&csr)
	}
	return csr
}

func newCSRPEM(t *testing.T) []byte {
	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			Organization: []string{"testing-org"},
		},
	}

	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, key)
	if err != nil {
		t.Fatal(err)
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csrDER,
	}

	p := pem.EncodeToMemory(csrPemBlock)
	if p == nil {
		t.Fatal("invalid pem block")
	}

	return p
}

func withApprovedCondition() func(*api.CertificateSigningRequest) {
	return func(csr *api.CertificateSigningRequest) {
		csr.Status.Conditions = append(csr.Status.Conditions, api.CertificateSigningRequestCondition{
			Type:   api.CertificateApproved,
			Status: core.ConditionTrue,
		})
	}
}

func withDeniedCondition() func(*api.CertificateSigningRequest) {
	return func(csr *api.CertificateSigningRequest) {
		csr.Status.Conditions = append(csr.Status.Conditions, api.CertificateSigningRequestCondition{
			Type:   api.CertificateDenied,
			Status: core.ConditionTrue,
		})
	}
}

func withFailedCondition() func(*api.CertificateSigningRequest) {
	return func(csr *api.CertificateSigningRequest) {
		csr.Status.Conditions = append(csr.Status.Conditions, api.CertificateSigningRequestCondition{
			Type:   api.CertificateFailed,
			Status: core.ConditionTrue,
		})
	}
}
