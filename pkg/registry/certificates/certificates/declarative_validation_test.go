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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
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
		"no conditions - valid": {
			input: makeValidCSR(),
		},
		"approved condition - valid": {
			input: makeValidCSR(withApprovedCondition()),
		},
		"denied condition - valid": {
			input: makeValidCSR(withDeniedCondition()),
		},
		"failed condition - valid": {
			input: makeValidCSR(withFailedCondition()),
		},
		"approved+failed conditions - valid": {
			input: makeValidCSR(withApprovedCondition(), withFailedCondition()),
		},
		"denied+failed conditions - valid": {
			input: makeValidCSR(withDeniedCondition(), withFailedCondition()),
		},
		"approved+denied conditions - invalid": {
			input: makeValidCSR(withApprovedCondition(), withDeniedCondition()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions"), nil, "").WithOrigin("zeroOrOneOf"),
			},
		},
		"denied+approved conditions - invalid": {
			input: makeValidCSR(withDeniedCondition(), withApprovedCondition()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions"), nil, "").WithOrigin("zeroOrOneOf"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				// We only need to test both gate enabled and disabled together, because
				// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
				// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)

				errs := Strategy.Validate(ctx, &tc.input)
				if gateVal {
					declarativeTakeoverErrs = errs
				} else {
					imperativeErrs = errs
				}
				// The errOutputMatcher is used to verify the output matches the expected errors in test cases.
				errOutputMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
				if len(tc.expectedErrs) > 0 {
					errOutputMatcher.Test(t, tc.expectedErrs, errs)
				} else if len(errs) != 0 {
					t.Errorf("expected no errors, but got: %v", errs)
				}
			}
			// The equivalenceMatcher is used to verify the output errors from hand-written imperative validation
			// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.input, nil)
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
				var declarativeTakeoverErrs field.ErrorList
				var imperativeErrs field.ErrorList
				for _, gateVal := range []bool{true, false} {
					// We only need to test both gate enabled and disabled together, because
					// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
					// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)
					errs := strategy.ValidateUpdate(ctx, &tc.update, &tc.old)
					if gateVal {
						declarativeTakeoverErrs = errs
					} else {
						imperativeErrs = errs
					}
					// The errOutputMatcher is used to verify the output matches the expected errors in test cases.
					errOutputMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()

					if len(tc.expectedErrs) > 0 {
						errOutputMatcher.Test(t, tc.expectedErrs, errs)
					} else if len(errs) != 0 {
						t.Errorf("expected no errors, but got: %v", errs)
					}
				}
				// The equivalenceMatcher is used to verify the output errors from hand-written imperative validation
				// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
				equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
				// TODO: remove this once ErrorMatcher has been extended to handle this form of deduplication.
				dedupedImperativeErrs := field.ErrorList{}
				for _, err := range imperativeErrs {
					found := false
					for _, existingErr := range dedupedImperativeErrs {
						if equivalenceMatcher.Matches(existingErr, err) {
							found = true
							break
						}
					}
					if !found {
						dedupedImperativeErrs = append(dedupedImperativeErrs, err)
					}
				}
				equivalenceMatcher.Test(t, dedupedImperativeErrs, declarativeTakeoverErrs)

				apitesting.VerifyVersionedValidationEquivalence(t, &tc.update, &tc.old)
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
