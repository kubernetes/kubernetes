/*
Copyright 2020 The Kubernetes Authors.

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
	"testing"

	capi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"

	"k8s.io/kubernetes/test/integration/framework"
)

// Verifies that the signerName field defaulting is wired up correctly.
// An exhaustive set of test cases for all permutations of the possible
// defaulting cases is written as a unit tests in the
// `pkg/apis/certificates/...` directory.
// This test cases exists to show that the defaulting function is wired up into
// the apiserver correctly.
func TestCSRSignerNameDefaulting(t *testing.T) {
	strPtr := func(s string) *string { return &s }
	tests := map[string]struct {
		csr                capi.CertificateSigningRequestSpec
		expectedSignerName string
	}{
		"defaults to legacy-unknown if not recognised": {
			csr: capi.CertificateSigningRequestSpec{
				Request: pemWithGroup(""),
				Usages:  []capi.KeyUsage{capi.UsageKeyEncipherment, capi.UsageDigitalSignature},
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default signerName if an explicit value is provided": {
			csr: capi.CertificateSigningRequestSpec{
				Request:    pemWithGroup(""),
				Usages:     []capi.KeyUsage{capi.UsageKeyEncipherment, capi.UsageDigitalSignature},
				SignerName: strPtr("example.com/my-custom-signer"),
			},
			expectedSignerName: "example.com/my-custom-signer",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, s, closeFn := framework.RunAMaster(nil)
			defer closeFn()
			client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			csrClient := client.CertificatesV1beta1().CertificateSigningRequests()
			csr := &capi.CertificateSigningRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "testcsr"},
				Spec:       test.csr,
			}
			csr, err := csrClient.Create(context.TODO(), csr, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create CSR resource: %v", err)
			}
			if *csr.Spec.SignerName != test.expectedSignerName {
				t.Errorf("expected CSR signerName to be %q but it was %q", test.expectedSignerName, *csr.Spec.SignerName)
			}
		})
	}
}
