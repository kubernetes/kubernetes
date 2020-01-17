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

package approver

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/rand"
	"net"
	"testing"

	authorization "k8s.io/api/authorization/v1"
	capi "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	testclient "k8s.io/client-go/testing"
	k8s_certificates_v1beta1 "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

func TestHasKubeletUsages(t *testing.T) {
	cases := []struct {
		usages   []capi.KeyUsage
		expected bool
	}{
		{
			usages:   nil,
			expected: false,
		},
		{
			usages:   []capi.KeyUsage{},
			expected: false,
		},
		{
			usages: []capi.KeyUsage{
				capi.UsageKeyEncipherment,
				capi.UsageDigitalSignature,
			},
			expected: false,
		},
		{
			usages: []capi.KeyUsage{
				capi.UsageKeyEncipherment,
				capi.UsageDigitalSignature,
				capi.UsageServerAuth,
			},
			expected: false,
		},
		{
			usages: []capi.KeyUsage{
				capi.UsageKeyEncipherment,
				capi.UsageDigitalSignature,
				capi.UsageClientAuth,
			},
			expected: true,
		},
	}
	for _, c := range cases {
		if hasExactUsages(&capi.CertificateSigningRequest{
			Spec: capi.CertificateSigningRequestSpec{
				Usages: c.usages,
			},
		}, kubeletClientUsages) != c.expected {
			t.Errorf("unexpected result of hasKubeletUsages(%v), expecting: %v", c.usages, c.expected)
		}
	}
}

func TestHandle(t *testing.T) {
	cases := []struct {
		allowed    bool
		recognized bool
		err        bool
		verify     func(*testing.T, []testclient.Action)
	}{
		{
			recognized: false,
			allowed:    false,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no client calls but got: %#v", as)
				}
			},
		},
		{
			recognized: false,
			allowed:    true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no client calls but got: %#v", as)
				}
			},
		},
		{
			recognized: true,
			allowed:    false,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected 1 call but got: %#v", as)
					return
				}
				_ = as[0].(testclient.CreateActionImpl)
			},
			err: true,
		},
		{
			recognized: true,
			allowed:    true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 2 {
					t.Errorf("expected two calls but got: %#v", as)
					return
				}
				_ = as[0].(testclient.CreateActionImpl)
				a := as[1].(testclient.UpdateActionImpl)
				if got, expected := a.Verb, "update"; got != expected {
					t.Errorf("got: %v, expected: %v", got, expected)
				}
				if got, expected := a.Resource, (schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}); got != expected {
					t.Errorf("got: %v, expected: %v", got, expected)
				}
				if got, expected := a.Subresource, "approval"; got != expected {
					t.Errorf("got: %v, expected: %v", got, expected)
				}
				csr := a.Object.(*capi.CertificateSigningRequest)
				if len(csr.Status.Conditions) != 1 {
					t.Errorf("expected CSR to have approved condition: %#v", csr)
				}
				c := csr.Status.Conditions[0]
				if got, expected := c.Type, capi.CertificateApproved; got != expected {
					t.Errorf("got: %v, expected: %v", got, expected)
				}
				if got, expected := c.Reason, "AutoApproved"; got != expected {
					t.Errorf("got: %v, expected: %v", got, expected)
				}
			},
		},
	}

	for _, c := range cases {
		t.Run(fmt.Sprintf("recognized:%v,allowed: %v,err: %v", c.recognized, c.allowed, c.err), func(t *testing.T) {
			client := &fake.Clientset{}
			client.AddReactor("create", "subjectaccessreviews", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
				return true, &authorization.SubjectAccessReview{
					Status: authorization.SubjectAccessReviewStatus{
						Allowed: c.allowed,
					},
				}, nil
			})
			approver := sarApprover{
				client: client,
				recognizers: []csrRecognizer{
					{
						successMessage: "tester",
						permission:     authorization.ResourceAttributes{Group: "foo", Resource: "bar", Subresource: "baz"},
						recognize: func(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool {
							return c.recognized
						},
					},
				},
			}
			csr := makeTestCsr()
			if err := approver.handle(csr); err != nil && !c.err {
				t.Errorf("unexpected err: %v", err)
			}
			c.verify(t, client.Actions())
		})
	}
}

func TestRecognizers(t *testing.T) {
	goodCases := []func(b *csrBuilder){
		func(b *csrBuilder) {
		},
	}

	testRecognizer(t, goodCases, isNodeClientCert, true)
	testRecognizer(t, goodCases, isSelfNodeClientCert, true)

	badCases := []func(b *csrBuilder){
		func(b *csrBuilder) {
			b.cn = "mike"
		},
		func(b *csrBuilder) {
			b.orgs = nil
		},
		func(b *csrBuilder) {
			b.orgs = []string{"system:master"}
		},
		func(b *csrBuilder) {
			b.usages = append(b.usages, capi.UsageServerAuth)
		},
	}

	testRecognizer(t, badCases, isNodeClientCert, false)
	testRecognizer(t, badCases, isSelfNodeClientCert, false)

	// cn different then requestor
	differentCN := []func(b *csrBuilder){
		func(b *csrBuilder) {
			b.requestor = "joe"
		},
		func(b *csrBuilder) {
			b.cn = "system:node:bar"
		},
	}

	testRecognizer(t, differentCN, isNodeClientCert, true)
	testRecognizer(t, differentCN, isSelfNodeClientCert, false)
}

func testRecognizer(t *testing.T, cases []func(b *csrBuilder), recognizeFunc func(csr *capi.CertificateSigningRequest, x509cr *x509.CertificateRequest) bool, shouldRecognize bool) {
	for _, c := range cases {
		b := csrBuilder{
			cn:        "system:node:foo",
			orgs:      []string{"system:nodes"},
			requestor: "system:node:foo",
			usages: []capi.KeyUsage{
				capi.UsageKeyEncipherment,
				capi.UsageDigitalSignature,
				capi.UsageClientAuth,
			},
		}
		c(&b)
		t.Run(fmt.Sprintf("csr:%#v", b), func(t *testing.T) {
			csr := makeFancyTestCsr(b)
			x509cr, err := k8s_certificates_v1beta1.ParseCSR(csr)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if recognizeFunc(csr, x509cr) != shouldRecognize {
				t.Errorf("expected recognized to be %v", shouldRecognize)
			}
		})
	}
}

// noncryptographic for faster testing
// DO NOT COPY THIS CODE
var insecureRand = rand.New(rand.NewSource(0))

func makeTestCsr() *capi.CertificateSigningRequest {
	return makeFancyTestCsr(csrBuilder{cn: "test-cert"})
}

type csrBuilder struct {
	cn        string
	orgs      []string
	requestor string
	usages    []capi.KeyUsage
	dns       []string
	emails    []string
	ips       []net.IP
}

func makeFancyTestCsr(b csrBuilder) *capi.CertificateSigningRequest {
	pk, err := ecdsa.GenerateKey(elliptic.P256(), insecureRand)
	if err != nil {
		panic(err)
	}
	csrb, err := x509.CreateCertificateRequest(insecureRand, &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   b.cn,
			Organization: b.orgs,
		},
		DNSNames:       b.dns,
		EmailAddresses: b.emails,
		IPAddresses:    b.ips,
	}, pk)
	if err != nil {
		panic(err)
	}
	return &capi.CertificateSigningRequest{
		Spec: capi.CertificateSigningRequestSpec{
			Username: b.requestor,
			Usages:   b.usages,
			Request:  pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE REQUEST", Bytes: csrb}),
		},
	}
}
