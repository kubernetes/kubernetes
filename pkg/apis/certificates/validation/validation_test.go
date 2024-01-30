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
	"math/big"
	mathrand "math/rand"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/util/certificate/csr"
	capi "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/pointer"
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
				field.Required(specPath.Child("usages"), ""),
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
				field.Required(specPath.Child("signerName"), ""),
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
		},
		"negative duration": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:            validUsages,
					Request:           newCSRPEM(t),
					SignerName:        validSignerName,
					ExpirationSeconds: pointer.Int32(-1),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("expirationSeconds"), int32(-1), "may not specify a duration less than 600 seconds (10 minutes)"),
			},
		},
		"zero duration": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:            validUsages,
					Request:           newCSRPEM(t),
					SignerName:        validSignerName,
					ExpirationSeconds: pointer.Int32(0),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("expirationSeconds"), int32(0), "may not specify a duration less than 600 seconds (10 minutes)"),
			},
		},
		"one duration": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:            validUsages,
					Request:           newCSRPEM(t),
					SignerName:        validSignerName,
					ExpirationSeconds: pointer.Int32(1),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("expirationSeconds"), int32(1), "may not specify a duration less than 600 seconds (10 minutes)"),
			},
		},
		"too short duration": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:            validUsages,
					Request:           newCSRPEM(t),
					SignerName:        validSignerName,
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Minute),
				},
			},
			errs: field.ErrorList{
				field.Invalid(specPath.Child("expirationSeconds"), *csr.DurationToExpirationSeconds(time.Minute), "may not specify a duration less than 600 seconds (10 minutes)"),
			},
		},
		"valid duration": {
			csr: capi.CertificateSigningRequest{
				ObjectMeta: validObjectMeta,
				Spec: capi.CertificateSigningRequestSpec{
					Usages:            validUsages,
					Request:           newCSRPEM(t),
					SignerName:        validSignerName,
					ExpirationSeconds: csr.DurationToExpirationSeconds(10 * time.Minute),
				},
			},
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
				field.Required(specPath.Child("usages"), ""),
			},
		},
		"unknown and duplicate usages": {
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
			el := ValidateCertificateSigningRequestCreate(&test.csr)
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
		name   string
		newCSR *capi.CertificateSigningRequest
		oldCSR *capi.CertificateSigningRequest
		want   certificateValidationOptions
	}{{
		name:   "strict create",
		oldCSR: nil,
		want:   certificateValidationOptions{},
	}, {
		name:   "strict update",
		oldCSR: &capi.CertificateSigningRequest{},
		want:   certificateValidationOptions{},
	}, {
		name: "compatible update, approved+denied",
		oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved}, {Type: capi.CertificateDenied}},
		}},
		want: certificateValidationOptions{
			allowBothApprovedAndDenied: true,
		},
	}, {
		name:   "compatible update, legacy signerName",
		oldCSR: &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{SignerName: capi.LegacyUnknownSignerName}},
		want: certificateValidationOptions{
			allowLegacySignerName: true,
		},
	}, {
		name: "compatible update, duplicate condition types",
		oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved}, {Type: capi.CertificateApproved}},
		}},
		want: certificateValidationOptions{
			allowDuplicateConditionTypes: true,
		},
	}, {
		name: "compatible update, empty condition types",
		oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{}},
		}},
		want: certificateValidationOptions{
			allowEmptyConditionType: true,
		},
	}, {
		name: "compatible update, no diff to certificate",
		newCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Certificate: validCertificate,
		}},
		oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Certificate: validCertificate,
		}},
		want: certificateValidationOptions{
			allowArbitraryCertificate: true,
		},
	}, {
		name: "compatible update, existing invalid certificate",
		newCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Certificate: []byte(`new - no PEM blocks`),
		}},
		oldCSR: &capi.CertificateSigningRequest{Status: capi.CertificateSigningRequestStatus{
			Certificate: []byte(`old - no PEM blocks`),
		}},
		want: certificateValidationOptions{
			allowArbitraryCertificate: true,
		},
	}, {
		name:   "compatible update, existing unknown usages",
		oldCSR: &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{Usages: []capi.KeyUsage{"unknown"}}},
		want: certificateValidationOptions{
			allowUnknownUsages: true,
		},
	}, {
		name:   "compatible update, existing duplicate usages",
		oldCSR: &capi.CertificateSigningRequest{Spec: capi.CertificateSigningRequestSpec{Usages: []capi.KeyUsage{"any", "any"}}},
		want: certificateValidationOptions{
			allowDuplicateUsages: true,
		},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getValidationOptions(tt.newCSR, tt.oldCSR); !reflect.DeepEqual(got, tt.want) {
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
		name   string
		newCSR *capi.CertificateSigningRequest
		oldCSR *capi.CertificateSigningRequest
		errs   []string
	}{{
		name:   "no-op",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
	}, {
		name:   "finalizer change with invalid status",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
	}, {
		name: "add Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.conditions: Forbidden: updates may not add a condition of type "Approved"`,
		},
	}, {
		name:   "remove Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`,
		},
	}, {
		name: "add Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.conditions: Forbidden: updates may not add a condition of type "Denied"`,
		},
	}, {
		name:   "remove Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`,
		},
	}, {
		name: "add Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs:   []string{},
	}, {
		name:   "remove Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`,
		},
	}, {
		name: "set certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: validCertificate,
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.certificate: Forbidden: updates may not set certificate content`,
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErrs := sets.NewString()
			for _, err := range ValidateCertificateSigningRequestUpdate(tt.newCSR, tt.oldCSR) {
				gotErrs.Insert(err.Error())
			}
			wantErrs := sets.NewString(tt.errs...)
			for _, missing := range wantErrs.Difference(gotErrs).List() {
				t.Errorf("missing expected error: %s", missing)
			}
			for _, unexpected := range gotErrs.Difference(wantErrs).List() {
				t.Errorf("unexpected error: %s", unexpected)
			}
		})
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
		name   string
		newCSR *capi.CertificateSigningRequest
		oldCSR *capi.CertificateSigningRequest
		errs   []string
	}{{
		name:   "no-op",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
	}, {
		name:   "finalizer change with invalid status",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
	}, {
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
	}, {
		name: "add Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.conditions: Forbidden: updates may not add a condition of type "Approved"`,
		},
	}, {
		name:   "remove Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`,
		},
	}, {
		name: "add Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.conditions: Forbidden: updates may not add a condition of type "Denied"`,
		},
	}, {
		name:   "remove Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`,
		},
	}, {
		name: "add Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs:   []string{},
	}, {
		name:   "remove Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`,
		},
	}, {
		name: "set valid certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: validCertificate,
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs:   []string{},
	}, {
		name: "set invalid certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: invalidCertificateNoPEM,
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.certificate: Invalid value: "<certificate data>": must contain at least one CERTIFICATE PEM block`,
		},
	}, {
		name: "reset certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: invalidCertificateNonCertificatePEM,
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: invalidCertificateNoPEM,
		}},
		errs: []string{
			`status.certificate: Forbidden: updates may not modify existing certificate content`,
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErrs := sets.NewString()
			for _, err := range ValidateCertificateSigningRequestStatusUpdate(tt.newCSR, tt.oldCSR) {
				gotErrs.Insert(err.Error())
			}
			wantErrs := sets.NewString(tt.errs...)
			for _, missing := range wantErrs.Difference(gotErrs).List() {
				t.Errorf("missing expected error: %s", missing)
			}
			for _, unexpected := range gotErrs.Difference(wantErrs).List() {
				t.Errorf("unexpected error: %s", unexpected)
			}
		})
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
		name   string
		newCSR *capi.CertificateSigningRequest
		oldCSR *capi.CertificateSigningRequest
		errs   []string
	}{{
		name:   "no-op",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
	}, {
		name:   "finalizer change with invalid certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{Certificate: invalidCertificateNoPEM}},
	}, {
		name: "add Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
	}, {
		name:   "remove Approved condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Approved"`,
		},
	}, {
		name: "add Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
	}, {
		name:   "remove Denied condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Denied"`,
		},
	}, {
		name: "add Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs:   []string{},
	}, {
		name:   "remove Failed condition",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
		}},
		errs: []string{
			`status.conditions: Forbidden: updates may not remove a condition of type "Failed"`,
		},
	}, {
		name: "set certificate",
		newCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMeta, Spec: validSpec, Status: capi.CertificateSigningRequestStatus{
			Certificate: validCertificate,
		}},
		oldCSR: &capi.CertificateSigningRequest{ObjectMeta: validUpdateMetaWithFinalizers, Spec: validSpec},
		errs: []string{
			`status.certificate: Forbidden: updates may not set certificate content`,
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErrs := sets.NewString()
			for _, err := range ValidateCertificateSigningRequestApprovalUpdate(tt.newCSR, tt.oldCSR) {
				gotErrs.Insert(err.Error())
			}
			wantErrs := sets.NewString(tt.errs...)
			for _, missing := range wantErrs.Difference(gotErrs).List() {
				t.Errorf("missing expected error: %s", missing)
			}
			for _, unexpected := range gotErrs.Difference(wantErrs).List() {
				t.Errorf("unexpected error: %s", unexpected)
			}
		})
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
		csr *capi.CertificateSigningRequest

		// options that allow the csr to pass validation
		lenientOpts certificateValidationOptions

		// regexes matching expected errors when validating strictly
		strictRegexes []regexp.Regexp

		// expected errors (after filtering out errors matched by strictRegexes) when validating strictly
		strictErrs []string
	}{
		// valid strict cases
		{
			name: "no status",
			csr:  &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec},
		}, {
			name: "approved condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
				},
			},
		}, {
			name: "denied condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateDenied, Status: core.ConditionTrue}},
				},
			},
		}, {
			name: "failed condition",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateFailed, Status: core.ConditionTrue}},
				},
			},
		}, {
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
		}, {
			name: "approved and denied",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions: []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}, {Type: capi.CertificateDenied, Status: core.ConditionTrue}},
				},
			},
			lenientOpts: certificateValidationOptions{allowBothApprovedAndDenied: true},
			strictErrs:  []string{`status.conditions[1].type: Invalid value: "Denied": Approved and Denied conditions are mutually exclusive`},
		}, {
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
		}, {
			name: "status.certificate, non-CERTIFICATE PEM",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonCertificatePEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": only CERTIFICATE PEM blocks are allowed, found "CERTIFICATE1"`},
		}, {
			name: "status.certificate, PEM headers",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificatePEMHeaders,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": no PEM block headers are permitted`},
		}, {
			name: "status.certificate, non-base64 PEM",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonBase64PEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": must contain at least one CERTIFICATE PEM block`},
		}, {
			name: "status.certificate, empty PEM block",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateEmptyPEM,
				},
			},
			lenientOpts: certificateValidationOptions{allowArbitraryCertificate: true},
			strictErrs:  []string{`status.certificate: Invalid value: "<certificate data>": found CERTIFICATE PEM block containing 0 certificates`},
		}, {
			name: "status.certificate, non-ASN1 data",
			csr: &capi.CertificateSigningRequest{ObjectMeta: validObjectMeta, Spec: validSpec,
				Status: capi.CertificateSigningRequestStatus{
					Conditions:  []capi.CertificateSigningRequestCondition{{Type: capi.CertificateApproved, Status: core.ConditionTrue}},
					Certificate: invalidCertificateNonASN1Data,
				},
			},
			lenientOpts:   certificateValidationOptions{allowArbitraryCertificate: true},
			strictRegexes: []regexp.Regexp{*regexp.MustCompile(`status.certificate: Invalid value: "\<certificate data\>": (asn1: structure error: sequence tag mismatch|x509: invalid RDNSequence)`)},
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

			// filter errors matching strictRegexes and ensure every strictRegex matches at least one error
			for _, expectedRegex := range tt.strictRegexes {
				matched := false
				for _, err := range gotErrs.List() {
					if expectedRegex.MatchString(err) {
						gotErrs.Delete(err)
						matched = true
					}
				}
				if !matched {
					t.Errorf("missing expected error matching: %s", expectedRegex.String())
				}
			}

			wantErrs := sets.NewString(tt.strictErrs...)
			for _, missing := range wantErrs.Difference(gotErrs).List() {
				t.Errorf("missing expected strict error: %s", missing)
			}
			for _, unexpected := range gotErrs.Difference(wantErrs).List() {
				t.Errorf("unexpected errors: %s", unexpected)
			}
		})
	}
}

func mustMakeCertificate(t *testing.T, template *x509.Certificate) []byte {
	gen := mathrand.New(mathrand.NewSource(12345))

	pub, priv, err := ed25519.GenerateKey(gen)
	if err != nil {
		t.Fatalf("Error while generating key: %v", err)
	}

	cert, err := x509.CreateCertificate(gen, template, template, pub, priv)
	if err != nil {
		t.Fatalf("Error while making certificate: %v", err)
	}

	return cert
}

func mustMakePEMBlock(blockType string, headers map[string]string, data []byte) string {
	return string(pem.EncodeToMemory(&pem.Block{
		Type:    blockType,
		Headers: headers,
		Bytes:   data,
	}))
}

func TestValidateClusterTrustBundle(t *testing.T) {
	goodCert1 := mustMakeCertificate(t, &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root1",
		},
		IsCA:                  true,
		BasicConstraintsValid: true,
	})

	goodCert2 := mustMakeCertificate(t, &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root2",
		},
		IsCA:                  true,
		BasicConstraintsValid: true,
	})

	badNotCACert := mustMakeCertificate(t, &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root3",
		},
	})

	goodCert1Block := string(mustMakePEMBlock("CERTIFICATE", nil, goodCert1))
	goodCert2Block := string(mustMakePEMBlock("CERTIFICATE", nil, goodCert2))

	goodCert1AlternateBlock := strings.ReplaceAll(goodCert1Block, "\n", "\n\t\n")

	badNotCACertBlock := string(mustMakePEMBlock("CERTIFICATE", nil, badNotCACert))

	badBlockHeadersBlock := string(mustMakePEMBlock("CERTIFICATE", map[string]string{"key": "value"}, goodCert1))
	badBlockTypeBlock := string(mustMakePEMBlock("NOTACERTIFICATE", nil, goodCert1))
	badNonParseableBlock := string(mustMakePEMBlock("CERTIFICATE", nil, []byte("this is not a certificate")))

	badTooBigBundle := ""
	for i := 0; i < (core.MaxSecretSize/len(goodCert1Block))+1; i++ {
		badTooBigBundle += goodCert1Block + "\n"
	}

	testCases := []struct {
		description string
		bundle      *capi.ClusterTrustBundle
		opts        ValidateClusterTrustBundleOptions
		wantErrors  field.ErrorList
	}{
		{
			description: "valid, no signer name",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block,
				},
			},
		},
		{
			description: "invalid, too big",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: badTooBigBundle,
				},
			},
			wantErrors: field.ErrorList{
				field.TooLong(field.NewPath("spec", "trustBundle"), fmt.Sprintf("<value omitted, len %d>", len(badTooBigBundle)), core.MaxSecretSize),
			},
		},
		{
			description: "invalid, no signer name, invalid name",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:bar:foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "k8s.io:bar:foo", "ClusterTrustBundle without signer name must not have \":\" in its name"),
			},
		}, {
			description: "valid, with signer name",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:foo:bar",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "k8s.io/foo",
					TrustBundle: goodCert1Block,
				},
			},
		}, {
			description: "invalid, with signer name, missing name prefix",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "look-ma-no-prefix",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "k8s.io/foo",
					TrustBundle: goodCert1Block,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "look-ma-no-prefix", "ClusterTrustBundle for signerName k8s.io/foo must be named with prefix k8s.io:foo:"),
			},
		}, {
			description: "invalid, with signer name, empty name suffix",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:foo:",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "k8s.io/foo",
					TrustBundle: goodCert1Block,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "k8s.io:foo:", `a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		}, {
			description: "invalid, with signer name, bad name suffix",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:foo:123notvalidDNSSubdomain",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "k8s.io/foo",
					TrustBundle: goodCert1Block,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "k8s.io:foo:123notvalidDNSSubdomain", `a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		}, {
			description: "valid, with signer name, with inter-block garbage",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:foo:abc",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "k8s.io/foo",
					TrustBundle: "garbage\n" + goodCert1Block + "\ngarbage\n" + goodCert2Block,
				},
			},
		}, {
			description: "invalid, no signer name, no trust anchors",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "at least one trust anchor must be provided"),
			},
		}, {
			description: "invalid, no trust anchors",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "k8s.io:foo:abc",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName: "k8s.io/foo",
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "at least one trust anchor must be provided"),
			},
		}, {
			description: "invalid, bad signer name",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "invalid:foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					SignerName:  "invalid",
					TrustBundle: goodCert1Block,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "signerName"), "invalid", "must be a fully qualified domain and path of the form 'example.com/signer-name'"),
			},
		}, {
			description: "invalid, no blocks",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: "non block garbage",
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "at least one trust anchor must be provided"),
			},
		}, {
			description: "invalid, bad block type",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block + "\n" + badBlockTypeBlock,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "entry 1 has bad block type: NOTACERTIFICATE"),
			},
		}, {
			description: "invalid, block with headers",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block + "\n" + badBlockHeadersBlock,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "entry 1 has PEM block headers"),
			},
		}, {
			description: "invalid, cert is not a CA cert",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: badNotCACertBlock,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "entry 0 does not have the CA bit set"),
			},
		}, {
			description: "invalid, duplicated blocks",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block + "\n" + goodCert1AlternateBlock,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "duplicate trust anchor (indices [0 1])"),
			},
		}, {
			description: "invalid, non-certificate entry",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: goodCert1Block + "\n" + badNonParseableBlock,
				},
			},
			wantErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "entry 1 does not parse as X.509"),
			},
		}, {
			description: "allow any old garbage in the PEM field if we suppress parsing",
			bundle: &capi.ClusterTrustBundle{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: capi.ClusterTrustBundleSpec{
					TrustBundle: "garbage",
				},
			},
			opts: ValidateClusterTrustBundleOptions{
				SuppressBundleParsing: true,
			},
		}}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			gotErrors := ValidateClusterTrustBundle(tc.bundle, tc.opts)
			if diff := cmp.Diff(gotErrors, tc.wantErrors); diff != "" {
				t.Fatalf("Unexpected error output from Validate; diff (-got +want)\n%s", diff)
			}

			// When there are no changes to the object,
			// ValidateClusterTrustBundleUpdate should not report errors about
			// the TrustBundle field.
			tc.bundle.ObjectMeta.ResourceVersion = "1"
			newBundle := tc.bundle.DeepCopy()
			newBundle.ObjectMeta.ResourceVersion = "2"
			gotErrors = ValidateClusterTrustBundleUpdate(newBundle, tc.bundle)

			var filteredWantErrors field.ErrorList
			for _, err := range tc.wantErrors {
				if err.Field != "spec.trustBundle" {
					filteredWantErrors = append(filteredWantErrors, err)
				}
			}

			if diff := cmp.Diff(gotErrors, filteredWantErrors); diff != "" {
				t.Fatalf("Unexpected error output from ValidateUpdate; diff (-got +want)\n%s", diff)
			}
		})
	}
}

func TestValidateClusterTrustBundleUpdate(t *testing.T) {
	goodCert1 := mustMakeCertificate(t, &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root1",
		},
		IsCA:                  true,
		BasicConstraintsValid: true,
	})

	goodCert2 := mustMakeCertificate(t, &x509.Certificate{
		SerialNumber: big.NewInt(0),
		Subject: pkix.Name{
			CommonName: "root2",
		},
		IsCA:                  true,
		BasicConstraintsValid: true,
	})

	goodCert1Block := string(mustMakePEMBlock("CERTIFICATE", nil, goodCert1))
	goodCert2Block := string(mustMakePEMBlock("CERTIFICATE", nil, goodCert2))

	testCases := []struct {
		description          string
		oldBundle, newBundle *capi.ClusterTrustBundle
		wantErrors           field.ErrorList
	}{{
		description: "changing signer name disallowed",
		oldBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: goodCert1Block,
			},
		},
		newBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/bar",
				TrustBundle: goodCert1Block,
			},
		},
		wantErrors: field.ErrorList{
			field.Invalid(field.NewPath("metadata", "name"), "k8s.io:foo:bar", "ClusterTrustBundle for signerName k8s.io/bar must be named with prefix k8s.io:bar:"),
			field.Invalid(field.NewPath("spec", "signerName"), "k8s.io/bar", "field is immutable"),
		},
	}, {
		description: "adding certificate allowed",
		oldBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: goodCert1Block,
			},
		},
		newBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: goodCert1Block + "\n" + goodCert2Block,
			},
		},
	}, {
		description: "emptying trustBundle disallowed",
		oldBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: goodCert1Block,
			},
		},
		newBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: "",
			},
		},
		wantErrors: field.ErrorList{
			field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "at least one trust anchor must be provided"),
		},
	}, {
		description: "emptying trustBundle (replace with non-block garbage) disallowed",
		oldBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: goodCert1Block,
			},
		},
		newBundle: &capi.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "k8s.io:foo:bar",
			},
			Spec: capi.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: "non block garbage",
			},
		},
		wantErrors: field.ErrorList{
			field.Invalid(field.NewPath("spec", "trustBundle"), "<value omitted>", "at least one trust anchor must be provided"),
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			tc.oldBundle.ObjectMeta.ResourceVersion = "1"
			tc.newBundle.ObjectMeta.ResourceVersion = "2"
			gotErrors := ValidateClusterTrustBundleUpdate(tc.newBundle, tc.oldBundle)
			if diff := cmp.Diff(gotErrors, tc.wantErrors); diff != "" {
				t.Errorf("Unexpected error output from ValidateUpdate; diff (-got +want)\n%s", diff)
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
