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

package certificatesigningrequest

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "certificates.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "certificatesigningrequests",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        certificates.CertificateSigningRequest
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidCSR(),
		},
		"invalid signerName (empty)": {
			input: mkValidCSR(func(obj *certificates.CertificateSigningRequest) {
				obj.Spec.SignerName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "signerName"), ""),
			},
		},
		"invalid usages (empty)": {
			input: mkValidCSR(func(obj *certificates.CertificateSigningRequest) {
				obj.Spec.Usages = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "usages"), ""),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func mkValidCSR(tweaks ...func(obj *certificates.CertificateSigningRequest)) certificates.CertificateSigningRequest {
	obj := certificates.CertificateSigningRequest{
		Spec: certificates.CertificateSigningRequestSpec{
			Request:    []byte("some-csr-data"),
			SignerName: "example.com/signer",
			Usages:     []certificates.KeyUsage{certificates.UsageDigitalSignature},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
