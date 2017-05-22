/*
Copyright 2017 The Kubernetes Authors.

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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"testing"

	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

func TestHasKubeletUsages(t *testing.T) {
	cases := []struct {
		usages   []certificates.KeyUsage
		expected bool
	}{
		{
			usages:   nil,
			expected: false,
		},
		{
			usages:   []certificates.KeyUsage{},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
			},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
				certificates.UsageServerAuth,
			},
			expected: false,
		},
		{
			usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
				certificates.UsageClientAuth,
			},
			expected: true,
		},
	}
	for _, c := range cases {
		if hasExactUsages(&certificates.CertificateSigningRequest{
			Spec: certificates.CertificateSigningRequestSpec{
				Usages: c.usages,
			},
		}, kubeletClientUsages) != c.expected {
			t.Errorf("unexpected result of hasKubeletUsages(%v), expecting: %v", c.usages, c.expected)
		}
	}
}

func TestAutoApprove(t *testing.T) {
	certificateRequestPEM, err := generateCertificateRequest("system:node:test-node-name", []string{"system:nodes"})
	if err != nil {
		t.Fatalf("Unable to generate a certificate request: %v", err)
	}

	workingCertificateSigningRequest := certificates.CertificateSigningRequest{
		Spec: certificates.CertificateSigningRequestSpec{
			Groups:  []string{"approved-group"},
			Request: certificateRequestPEM,
			Usages: []certificates.KeyUsage{
				certificates.UsageKeyEncipherment,
				certificates.UsageDigitalSignature,
				certificates.UsageClientAuth,
			},
		},
		Status: certificates.CertificateSigningRequestStatus{
			Conditions: []certificates.CertificateSigningRequestCondition{},
		},
	}
	testCases := []struct {
		description                string
		approvalGroup              string
		csr                        certificates.CertificateSigningRequest
		expectError                bool
		expectedApproved           bool
		expectedDenied             bool
		expectedNumberOfConditions int
	}{{
		description:                "Matching approval group",
		approvalGroup:              "approved-group",
		csr:                        workingCertificateSigningRequest,
		expectError:                false,
		expectedApproved:           true,
		expectedDenied:             false,
		expectedNumberOfConditions: 1,
	}}
	successTestCase := testCases[0]

	tc := successTestCase
	tc.description = "No approval group configured"
	tc.approvalGroup = ""
	tc.expectedApproved = false
	tc.expectedNumberOfConditions = 0
	testCases = append(testCases, tc)

	tc = successTestCase
	tc.description = "Already approved certificate"
	tc.csr.Spec.Usages = []certificates.KeyUsage{}
	tc.csr.Status = certificates.CertificateSigningRequestStatus{
		Conditions: []certificates.CertificateSigningRequestCondition{
			{
				Type:    certificates.CertificateApproved,
				Message: "Test case mock approved.",
			},
		},
	}
	testCases = append(testCases, tc)

	tc = successTestCase
	tc.description = "Already denied certificate"
	tc.csr.Status = certificates.CertificateSigningRequestStatus{
		Conditions: []certificates.CertificateSigningRequestCondition{
			{
				Type:    certificates.CertificateDenied,
				Message: "Test case mock denied.",
			},
		},
	}
	tc.expectedApproved = false
	tc.expectedDenied = true
	testCases = append(testCases, tc)

	tc = successTestCase
	tc.description = "No matching approval group"
	tc.csr.Spec.Groups = []string{"not-approval-group"}
	tc.expectedApproved = false
	tc.expectedNumberOfConditions = 0
	testCases = append(testCases, tc)

	tc = successTestCase
	tc.description = "Certificate requested usages don't match allowed usages."
	tc.csr.Spec.Usages = []certificates.KeyUsage{
		certificates.UsageKeyEncipherment,
	}
	tc.expectedApproved = false
	tc.expectedNumberOfConditions = 0
	testCases = append(testCases, tc)

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			ga := NewGroupApprover(tc.approvalGroup)
			csr, err := ga.AutoApprove(&tc.csr)
			if tc.expectError && err == nil {
				t.Error("Got no error, wanted an error.")
			} else if !tc.expectError && err != nil {
				t.Errorf("Got error %v, wanted no error.", err)
			}
			approved, denied := getCertApprovalCondition(&csr.Status)
			if approved != tc.expectedApproved {
				t.Errorf("Got approved status %t, wanted %t", approved, tc.expectedApproved)
			}
			if denied != tc.expectedDenied {
				t.Errorf("Got denied status %t, wanted %t", denied, tc.expectedDenied)
			}
			if len(csr.Status.Conditions) != tc.expectedNumberOfConditions {
				t.Errorf("Got %d conditions, wanted %d conditions", len(csr.Status.Conditions), tc.expectedNumberOfConditions)
			}
		})
	}
}

func generateCertificateRequest(commonName string, organization []string) ([]byte, error) {
	ecdsa384Priv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, err
	}
	certificateRequestDER, err := x509.CreateCertificateRequest(
		rand.Reader,
		&x509.CertificateRequest{
			Subject: pkix.Name{
				CommonName:   commonName,
				Organization: organization,
			},
			SignatureAlgorithm: x509.ECDSAWithSHA1,
		},
		ecdsa384Priv)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{
		Type:    "CERTIFICATE REQUEST",
		Headers: map[string]string{},
		Bytes:   certificateRequestDER,
	}), nil
}
