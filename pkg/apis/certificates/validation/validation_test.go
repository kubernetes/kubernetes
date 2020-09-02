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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/certificates"
	capi "k8s.io/kubernetes/pkg/apis/certificates"
	capiv1beta1 "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/apis/core"
)

var (
	validObjectMeta = metav1.ObjectMeta{Name: "testcsr"}
	validSignerName = "example.com/valid-name"
	validUsages     = []capi.KeyUsage{capi.UsageKeyEncipherment}
)

func TestValidateCertificateSigningRequestCreate(t *testing.T) {
	specPath := field.NewPath("spec")
	// maxLengthSignerName is a signerName that is of maximum length, utilising
	// the max length specifications defined in validation.go.
	// It is of the form <fqdn(253)>/<resource-namespace(63)>.<resource-name(253)>
	maxLengthFQDN := fmt.Sprintf("%s.%s.%s.%s", repeatString("a", 63), repeatString("a", 63), repeatString("a", 63), repeatString("a", 61))
	maxLengthSignerName := fmt.Sprintf("%s/%s.%s", maxLengthFQDN, repeatString("a", 63), repeatString("a", 253))
	tests := map[string]struct {
		csr  capi.CertificateSigningRequest
		gv   schema.GroupVersion
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
				field.Invalid(specPath.Child("signerName"), "", `validating label "": a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
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
				field.Invalid(specPath.Child("signerName"), "something-", `validating label "something-": a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
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
				field.Invalid(specPath.Child("signerName"), "-", `validating label "-": a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
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
		"missing usages": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     []capi.KeyUsage{},
					Request:    newCSRPEM(t),
					SignerName: validSignerName,
				},
			},
			errs: field.ErrorList{
				field.Required(specPath.Child("usages"), "usages must be provided"),
			},
		},
		"unknown and duplicate usages - v1beta1": {
			gv: schema.GroupVersion{Group: capi.SchemeGroupVersion.Group, Version: "v1beta1"},
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     []capi.KeyUsage{"unknown", "unknown"},
					Request:    newCSRPEM(t),
					SignerName: validSignerName,
				},
			},
			errs: field.ErrorList{},
		},
		"unknown and duplicate usages - v1": {
			gv: schema.GroupVersion{Group: capi.SchemeGroupVersion.Group, Version: "v1"},
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:     []capi.KeyUsage{"unknown", "unknown"},
					Request:    newCSRPEM(t),
					SignerName: validSignerName,
				},
			},
			errs: field.ErrorList{
				field.NotSupported(specPath.Child("usages").Index(0), capi.KeyUsage("unknown"), allValidUsages.List()),
				field.NotSupported(specPath.Child("usages").Index(1), capi.KeyUsage("unknown"), allValidUsages.List()),
				field.Duplicate(specPath.Child("usages").Index(1), capi.KeyUsage("unknown")),
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			el := ValidateCertificateSigningRequestCreate(&test.csr, test.gv)
			if !reflect.DeepEqual(el, test.errs) {
				t.Errorf("returned and expected errors did not match - expected\n%v\nbut got\n%v", test.errs.ToAggregate(), el.ToAggregate())
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

func Test_getValidationOptions(t *testing.T) {
	tests := []struct {
		name    string
		version schema.GroupVersion
		newCSR  *certificates.CertificateSigningRequest
		oldCSR  *certificates.CertificateSigningRequest
		want    certificateValidationOptions
	}{
		{
			name:    "v1beta1 compatible create",
			version: capiv1beta1.SchemeGroupVersion,
			oldCSR:  nil,
			want: certificateValidationOptions{
				allowResettingCertificate:    true,
				allowBothApprovedAndDenied:   false,
				allowLegacySignerName:        true,
				allowDuplicateConditionTypes: true,
				allowEmptyConditionType:      true,
				allowArbitraryCertificate:    true,
				allowUnknownUsages:           true,
				allowDuplicateUsages:         true,
			},
		},
		{
			name:    "v1 strict create",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR:  nil,
			want:    certificateValidationOptions{},
		},
		{
			name:    "v1beta1 compatible update",
			version: capiv1beta1.SchemeGroupVersion,
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved}, {Type: capi.CertificateDenied}},
			}},
			want: certificateValidationOptions{
				allowResettingCertificate:    true,
				allowBothApprovedAndDenied:   true, // existing object has both approved and denied
				allowLegacySignerName:        true,
				allowDuplicateConditionTypes: true,
				allowEmptyConditionType:      true,
				allowArbitraryCertificate:    true,
				allowUnknownUsages:           true,
				allowDuplicateUsages:         true,
			},
		},
		{
			name:    "v1 strict update",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR:  &capi.CertificateSigningRequest{},
			want:    certificateValidationOptions{},
		},
		{
			name:    "v1 compatible update, approved+denied",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved}, {Type: capi.CertificateDenied}},
			}},
			want: certificateValidationOptions{
				allowBothApprovedAndDenied: true,
			},
		},
		{
			name:    "v1 compatible update, legacy signerName",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR:  &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{SignerName: capi.LegacyUnknownSignerName}},
			want: certificateValidationOptions{
				allowLegacySignerName: true,
			},
		},
		{
			name:    "v1 compatible update, duplicate condition types",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved}, {Type: capi.CertificateApproved}},
			}},
			want: certificateValidationOptions{
				allowDuplicateConditionTypes: true,
			},
		},
		{
			name:    "v1 compatible update, empty condition types",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{}},
			}},
			want: certificateValidationOptions{
				allowEmptyConditionType: true,
			},
		},
		{
			name:    "v1 compatible update, no diff to certificate",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			newCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Certificate: validCertificate,
			}},
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Certificate: validCertificate,
			}},
			want: certificateValidationOptions{
				allowArbitraryCertificate: true,
			},
		},
		{
			name:    "v1 compatible update, existing invalid certificate",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			newCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Certificate: []byte(`new - no PEM blocks`),
			}},
			oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
				Certificate: []byte(`old - no PEM blocks`),
			}},
			want: certificateValidationOptions{
				allowArbitraryCertificate: true,
			},
		},
		{
			name:    "v1 compatible update, existing unknown usages",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR:  &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{Usages: []capi.KeyUsage{"unknown"}}},
			want: certificateValidationOptions{
				allowUnknownUsages: true,
			},
		},
		{
			name:    "v1 compatible update, existing duplicate usages",
			version: schema.GroupVersion{Group: "certificates.k8s.io", Version: "v1"},
			oldCSR:  &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{Usages: []capi.KeyUsage{"any", "any"}}},
			want: certificateValidationOptions{
				allowDuplicateUsages: true,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getValidationOptions(tt.version, tt.newCSR, tt.oldCSR); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got  %#v\nwant %#v", got, tt.want)
			}
		})
	}
}

func TestValidateCertificateSigningRequestUpdate(t *testing.T) {
	validUpdateMeta := validObjectMeta
	validUpdateMeta.ResourceVersion = "1"

	validUpdateMetaWithFinalizers := validUpdateMeta
	validUpdateMetaWithFinalizers.Finalizers = []string{"foo"}

	validSpec := capi.CertificateSigningRequestSpec{
		Usages:     validUsages,
		Request:    newCSRPEM(t),
		SignerName: "example.com/something",
	}

	tests := []struct {
		name        string
		newCSR      *certificates.CertificateSigningRequest
		oldCSR      *certificates.CertificateSigningRequest
		versionErrs map[string][]string
	}{
		{
			name:   "no-op",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		},
		{
			name:   "finalizer change with invalid status",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		},
		{
			name: "add Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not add a condition of type "Approved"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not add a condition of type "Approved"`},
			},
		},
		{
			name:   "remove Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
			},
		},
		{
			name: "add Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not add a condition of type "Denied"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not add a condition of type "Denied"`},
			},
		},
		{
			name:   "remove Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
			},
		},
		{
			name: "add Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			oldCSR:      &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{},
		},
		{
			name:   "remove Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
			},
		},
		{
			name: "set certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: validCertificate,
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.certificate: Forbidden: updates may not set certificate content`},
				"v1beta1": {`status.certificate: Forbidden: updates may not set certificate content`},
			},
		},
	}

	for _, tt := range tests {
		for _, version := range []string{"v1", "v1beta1"} {
			t.Run(tt.name+"_"+version, func(t *testing.T) {
				gotErrs := sets.NewString()
				for _, err := range ValidateCertificateSigningRequestUpdate(tt.newCSR, tt.oldCSR, schema.GroupVersion{Group: certificates.GroupName, Version: version}) {
					gotErrs.Insert(err.Error())
				}
				wantErrs := sets.NewString(tt.versionErrs[version]...)
				for _, missing := range wantErrs.Difference(gotErrs).List() {
					t.Errorf("missing expected error: %s", missing)
				}
				for _, unexpected := range gotErrs.Difference(wantErrs).List() {
					t.Errorf("unexpected error: %s", unexpected)
				}
			})
		}
	}
}

func TestValidateCertificateSigningRequestStatusUpdate(t *testing.T) {
	validUpdateMeta := validObjectMeta
	validUpdateMeta.ResourceVersion = "1"

	validUpdateMetaWithFinalizers := validUpdateMeta
	validUpdateMetaWithFinalizers.Finalizers = []string{"foo"}

	validSpec := capi.CertificateSigningRequestSpec{
		Usages:     validUsages,
		Request:    newCSRPEM(t),
		SignerName: "example.com/something",
	}

	tests := []struct {
		name        string
		newCSR      *certificates.CertificateSigningRequest
		oldCSR      *certificates.CertificateSigningRequest
		versionErrs map[string][]string
	}{
		{
			name:   "no-op",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		},
		{
			name:   "finalizer change with invalid status",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		},
		{
			name: "finalizer change with duplicate and unknown usages",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: capi.CertificateSigningRequestSpec{
				Usages:     []capi.KeyUsage{"unknown", "unknown"},
				Request:    newCSRPEM(t),
				SignerName: validSignerName,
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: capi.CertificateSigningRequestSpec{
				Usages:     []capi.KeyUsage{"unknown", "unknown"},
				Request:    newCSRPEM(t),
				SignerName: validSignerName,
			}},
		},
		{
			name: "add Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not add a condition of type "Approved"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not add a condition of type "Approved"`},
			},
		},
		{
			name:   "remove Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
			},
		},
		{
			name: "add Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not add a condition of type "Denied"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not add a condition of type "Denied"`},
			},
		},
		{
			name:   "remove Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
			},
		},
		{
			name: "add Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			oldCSR:      &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{},
		},
		{
			name:   "remove Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
			},
		},
		{
			name: "set valid certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: validCertificate,
			}},
			oldCSR:      &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{},
		},
		{
			name: "set invalid certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: invalidCertificateNoPEM,
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1": {`status.certificate: Invalid value: "<certificate data>": must contain at least one CERTIFICATE PEM block`},
			},
		},
		{
			name: "reset certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: invalidCertificateNonCertificatePEM,
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: invalidCertificateNoPEM,
			}},
			versionErrs: map[string][]string{
				"v1": {`status.certificate: Forbidden: updates may not modify existing certificate content`},
			},
		},
	}

	for _, tt := range tests {
		for _, version := range []string{"v1", "v1beta1"} {
			t.Run(tt.name+"_"+version, func(t *testing.T) {
				gotErrs := sets.NewString()
				for _, err := range ValidateCertificateSigningRequestStatusUpdate(tt.newCSR, tt.oldCSR, schema.GroupVersion{Group: certificates.GroupName, Version: version}) {
					gotErrs.Insert(err.Error())
				}
				wantErrs := sets.NewString(tt.versionErrs[version]...)
				for _, missing := range wantErrs.Difference(gotErrs).List() {
					t.Errorf("missing expected error: %s", missing)
				}
				for _, unexpected := range gotErrs.Difference(wantErrs).List() {
					t.Errorf("unexpected error: %s", unexpected)
				}
			})
		}
	}
}

func TestValidateCertificateSigningRequestApprovalUpdate(t *testing.T) {
	validUpdateMeta := validObjectMeta
	validUpdateMeta.ResourceVersion = "1"

	validUpdateMetaWithFinalizers := validUpdateMeta
	validUpdateMetaWithFinalizers.Finalizers = []string{"foo"}

	validSpec := capi.CertificateSigningRequestSpec{
		Usages:     validUsages,
		Request:    newCSRPEM(t),
		SignerName: "example.com/something",
	}

	tests := []struct {
		name        string
		newCSR      *certificates.CertificateSigningRequest
		oldCSR      *certificates.CertificateSigningRequest
		versionErrs map[string][]string
	}{
		{
			name:   "no-op",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		},
		{
			name:   "finalizer change with invalid certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		},
		{
			name: "add Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		},
		{
			name:   "remove Approved condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`},
			},
		},
		{
			name: "add Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		},
		{
			name:   "remove Denied condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`},
			},
		},
		{
			name: "add Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			oldCSR:      &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{},
		},
		{
			name:   "remove Failed condition",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
			}},
			versionErrs: map[string][]string{
				"v1":      {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
				"v1beta1": {`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`},
			},
		},
		{
			name: "set certificate",
			newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
				Certificate: validCertificate,
			}},
			oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
			versionErrs: map[string][]string{
				"v1":      {`status.certificate: Forbidden: updates may not set certificate content`},
				"v1beta1": {`status.certificate: Forbidden: updates may not set certificate content`},
			},
		},
	}

	for _, tt := range tests {
		for _, version := range []string{"v1", "v1beta1"} {
			t.Run(tt.name+"_"+version, func(t *testing.T) {
				gotErrs := sets.NewString()
				for _, err := range ValidateCertificateSigningRequestApprovalUpdate(tt.newCSR, tt.oldCSR, schema.GroupVersion{Group: certificates.GroupName, Version: version}) {
					gotErrs.Insert(err.Error())
				}
				wantErrs := sets.NewString(tt.versionErrs[version]...)
				for _, missing := range wantErrs.Difference(gotErrs).List() {
					t.Errorf("missing expected error: %s", missing)
				}
				for _, unexpected := range gotErrs.Difference(wantErrs).List() {
					t.Errorf("unexpected error: %s", unexpected)
				}
			})
		}
	}
}

// Test_validateCertificateSigningRequestOptions verifies validation options are effective in tolerating specific aspects of CSRs
func Test_validateCertificateSigningRequestOptions(t *testing.T) {
	validSpec := capi.CertificateSigningRequestSpec{
		Usages:     validUsages,
		Request:    newCSRPEM(t),
		SignerName: "example.com/something",
	}

	tests := []struct {
		// testcase name
		name string

		// csr being validated
		csr *certificates.CertificateSigningRequest

		// options that allow the csr to pass validation
		lenientOpts certificateValidationOptions

		// expected errors when validating strictly
		strictErrs []string
	}{
		// valid strict cases
		{
			name: "no status",
			csr:  &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec},
		},
		{
			name: "approved condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
				},
			},
		},
		{
			name: "denied condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
				},
			},
		},
		{
			name: "failed condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
				},
			},
		},
		{
			name: "approved+issued",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: validCertificate,
				},
			},
		},

		// legacy signer
		{
			name: "legacy signer",
			csr: &capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: func() capi.CertificateSigningRequestSpec {
					specCopy := validSpec
					specCopy.SignerName = capi.LegacyUnknownSignerName
					return specCopy
				}(),
			},
			lenientOpts: certificateValidationOptions{allowLegacySignerName: true},
			strictErrs:  []string{`spec.signerName: Invalid value: "kubernetes.io/legacy-unknown": the legacy signerName is not allowed via this API version`},
		},

		// invalid condition cases
		{
			name: "empty condition type",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Status: core.ConditionTrue}},
				},
			},
			lenientOpts: certificateValidationOptions{allowEmptyConditionType: true},
			strictErrs:  []string{`status.conditions[0].type: Required value`},
		},
		{
			name: "approved and denied",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}, {Type: capi.CertificateDenied, Status: core.ConditionTrue}},
				},
			},
			lenientOpts: certificateValidationOptions{allowBothApprovedAndDenied: true},
			strictErrs:  []string{`status.conditions[1].type: Invalid value: "Denied": Approved and Denied conditions are mutually exclusive`},
		},
		{
			name: "duplicate condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}, {Type: capi.CertificateApproved, Status: core.ConditionTrue}},
				},
			},
			lenientOpts: certificateValidationOptions{allowDuplicateConditionTypes: true},
			strictErrs:  []string{`status.conditions[1].type: Duplicate value: "Approved"`},
		},

		// invalid allowArbitraryCertificate cases
		{
			name: "status.certificate, no PEM",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNoPEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": must contain at least one CERTIFICATE PEM block`},
		},
		{
			name: "status.certificate, non-CERTIFICATE PEM",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonCertificatePEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": only CERTIFICATE PEM blocks are allowed, found "CERTIFICATE1"`},
		},
		{
			name: "status.certificate, PEM headers",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificatePEMHeaders,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": no PEM block headers are permitted`},
		},
		{
			name: "status.certificate, non-base64 PEM",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonBase64PEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": must contain at least one CERTIFICATE PEM block`},
		},
		{
			name: "status.certificate, empty PEM block",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateEmptyPEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": found CERTIFICATE PEM block containing 0 certificates`},
		},
		{
			name: "status.certificate, non-ASN1 data",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonASN1Data,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": asn1: structure error: sequence tag mismatch`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// make sure the lenient options validate with no errors
			for _, err := range validateCertificateSigningRequest(tt.csr, tt.lenientOpts) {
				t.Errorf("unexpected error with lenient options: %s", err.Error())
			}

			// make sure the strict options produce the expected errors
			gotErrs := sets.NewString()
			for _, err := range validateCertificateSigningRequest(tt.csr, certificateValidationOptions{}) {
				gotErrs.Insert(err.Error())
			}
			wantErrs := sets.NewString(tt.strictErrs...)
			for _, missing := range wantErrs.Difference(gotErrs).List() {
				t.Errorf("missing expected strict error: %s", missing)
			}
			for _, unexpected := range gotErrs.Difference(wantErrs).List() {
				t.Errorf("unexpected strict error: %s", unexpected)
			}
		})
	}
}

var (
	validCertificate = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----
Intermediate non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU6gAwIBAgIUfqZtjoFgczZ+oQZbEC/BDSS2J6wwCgYIKoZIzj0EAwIw
EjEQMA4GA1UEAxMHUm9vdC1DQTAgFw0xNjEwMTEwNTA2MDBaGA8yMTE2MDkxNzA1
MDYwMFowGjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMFkwEwYHKoZIzj0CAQYI
KoZIzj0DAQcDQgAEyWHEMMCctJg8Xa5YWLqaCPbk3MjB+uvXac42JM9pj4k9jedD
kpUJRkWIPzgJI8Zk/3cSzluUTixP6JBSDKtwwaN4MHYwDgYDVR0PAQH/BAQDAgGm
MBMGA1UdJQQMMAoGCCsGAQUFBwMCMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0OBBYE
FF+p0JcY31pz+mjNZnjv0Gum92vZMB8GA1UdIwQYMBaAFB7P6+i4/pfNjqZgJv/b
dgA7Fe4tMAoGCCqGSM49BAMCA0gAMEUCIQCTT1YWQZaAqfQ2oBxzOkJE2BqLFxhz
3smQlrZ5gCHddwIgcvT7puhYOzAgcvMn9+SZ1JOyZ7edODjshCVCRnuHK2c=
-----END CERTIFICATE-----
Trailing non-PEM content
`)

	invalidCertificateNoPEM = []byte(`no PEM content`)

	invalidCertificateNonCertificatePEM = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE1-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE1-----
Trailing non-PEM content
`)

	invalidCertificatePEMHeaders = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
Some-Header: Some-Value
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----
Trailing non-PEM content
`)

	invalidCertificateNonBase64PEM = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1d?????????
-----END CERTIFICATE-----
Trailing non-PEM content
`)

	invalidCertificateEmptyPEM = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
-----END CERTIFICATE-----
Trailing non-PEM content
`)

	// first character is invalid
	invalidCertificateNonASN1Data = []byte(`
Leading non-PEM content
-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUNRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----
Trailing non-PEM content
`)
)
