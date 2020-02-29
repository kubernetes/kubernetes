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

package validation

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	capi "k8s.io/kubernetes/pkg/apis/certificates"
)

var (
	validObjectMeta = metav1.ObjectMeta{Name: "testcsr"}
	validSignerName = "example.com/valid-name"
	validUsages     = []capi.KeyUsage{capi.UsageKeyEncipherment}
)

func TestValidateCertificateSigningRequest(t *testing.T) {
	specPath := field.NewPath("spec")
	// maxLengthSignerName is a signerName that is of maximum length, utilising
	// the max length specifications defined in validation.go.
	// It is of the form <fqdn(253)>/<resource-namespace(63)>.<resource-name(253)>
	maxLengthFQDN := fmt.Sprintf("%s.%s.%s.%s", repeatString("a", 63), repeatString("a", 63), repeatString("a", 63), repeatString("a", 61))
	maxLengthSignerName := fmt.Sprintf("%s/%s.%s", maxLengthFQDN, repeatString("a", 63), repeatString("a", 253))
	tests := map[string]struct {
		csr  capi.CertificateSigningRequest
		errs field.ErrorList
	}{
		"CSR with empty request data should fail": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					SignerName: validSignerName,
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("request"), []byte(nil), "PEM block type must be CERTIFICATE REQUEST"),
			},
		},
		"CSR with invalid request data should fail": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					SignerName: validSignerName,
					Request:    []byte("invalid data"),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("request"), []byte("invalid data"), "PEM block type must be CERTIFICATE REQUEST"),
			},
		},
		"CSR with no usages should fail": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					SignerName: validSignerName,
					Request:    newCSRPEM(t),
				},
			},
			errs: field.ErrorList{
				field.Required(specPath.Child("usages"), "usages must be provided"),
			},
		},
		"CSR with no signerName set should fail": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:  validUsages,
					Request: newCSRPEM(t),
				},
			},
			errs: field.ErrorList{
				field.Required(specPath.Child("signerName"), "signerName must be provided"),
			},
		},
		"signerName contains no '/'": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "an-invalid-signer-name",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "an-invalid-signer-name", "must be a fully qualified domain and path of the form 'example.com/signer-name'"),
			},
		},
		"signerName contains two '/'": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "an-invalid-signer-name.com/something/else",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "an-invalid-signer-name.com/something/else", "must be a fully qualified domain and path of the form 'example.com/signer-name'"),
			},
		},
		"signerName domain component is not fully qualified": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "example/some-signer-name",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "example", "should be a domain with at least two segments separated by dots"),
			},
		},
		"signerName path component is empty": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "example.com/",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "", `validating label "": a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		},
		"signerName path component ends with a symbol": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "example.com/something-",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "something-", `validating label "something-": a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		},
		"signerName path component is a symbol": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "example.com/-",
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), "-", `validating label "-": a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		},
		"signerName path component contains no '.' but is valid": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: "example.com/some-signer-name",
				},
			},
			errs: field.ErrorList{},
		},
		"signerName with a total length greater than 571 characters should be rejected": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:  validUsages,
					Request: newCSRPEM(t),
					// this string is longer than the max signerName limit (635 chars)
					SignerName: maxLengthSignerName + ".toolong",
				},
			},
			errs: field.ErrorList{
				field.TooLong(specPath.Child("signerName"), maxLengthSignerName+".toolong", len(maxLengthSignerName)),
			},
		},
		"signerName with a fqdn greater than 253 characters should be rejected": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:  validUsages,
					Request: newCSRPEM(t),
					// this string is longer than the max signerName limit (635 chars)
					SignerName: fmt.Sprintf("%s.extra/valid-path", maxLengthFQDN),
				},
			},
			errs: field.ErrorList{
				field.TooLong(specPath.Child("signerName"), fmt.Sprintf("%s.extra", maxLengthFQDN), len(maxLengthFQDN)),
			},
		},
		"signerName can have a longer path if the domain component is less than the max length": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: fmt.Sprintf("abc.io/%s.%s", repeatString("a", 253), repeatString("a", 253)),
				},
			},
			errs: field.ErrorList{},
		},
		"signerName with a domain label greater than 63 characters will fail": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: fmt.Sprintf("%s.example.io/valid-path", repeatString("a", 66)),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("signerName"), fmt.Sprintf("%s.example.io", repeatString("a", 66)), fmt.Sprintf(`validating label "%s": must be no more than 63 characters`, repeatString("a", 66))),
			},
		},
		"signerName of max length in format <fully-qualified-domain-name>/<resource-namespace>.<resource-name> is valid": {
			// ensure signerName is of the form domain.com/something and up to 571 characters.
			// This length and format is specified to accommodate signerNames like:
			// <fqdn>/<resource-namespace>.<resource-name>.
			// The max length of a FQDN is 253 characters (DNS1123Subdomain max length)
			// The max length of a namespace name is 63 characters (DNS1123Label max length)
			// The max length of a resource name is 253 characters (DNS1123Subdomain max length)
			// We then add an additional 2 characters to account for the one '.' and one '/'.
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     validUsages,
					Request:    newCSRPEM(t),
					SignerName: maxLengthSignerName,
				},
			},
			errs: field.ErrorList{},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			el := ValidateCertificateSigningRequest(&test.csr)
			if !reflect.DeepEqual(el, test.errs) {
				t.Errorf("returned and expected errors did not match - expected %v but got %v", test.errs.ToAggregate(), el.ToAggregate())
			}
		})
	}
}

func repeatString(s string, num int) string {
	l := make([]string, num)
	for i := 0; i < num; i++ {
		l[i] = s
	}
	return strings.Join(l, "")
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
