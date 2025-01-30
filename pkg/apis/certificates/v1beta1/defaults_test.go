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

package v1beta1

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"net"
	"net/url"
	"reflect"
	"testing"

	capi "k8s.io/api/certificates/v1beta1"
)

func TestIsKubeletServingCSR(t *testing.T) {
	newCSR := func(base pemOptions, overlays ...pemOptions) *x509.CertificateRequest {
		b := csrWithOpts(base, overlays...)
		csr, err := ParseCSR(b)
		if err != nil {
			t.Fatal(err)
		}
		return csr
	}
	tests := map[string]struct {
		req    *x509.CertificateRequest
		usages []capi.KeyUsage
		exp    bool
	}{
		"defaults for kubelet-serving": {
			req:    newCSR(kubeletServerPEMOptions),
			usages: kubeletServerUsages,
			exp:    true,
		},
		"defaults without key encipherment for kubelet-serving": {
			req:    newCSR(kubeletServerPEMOptions),
			usages: kubeletServerUsagesNoRSA,
			exp:    true,
		},
		"does not default to kube-apiserver-client-kubelet if org is not 'system:nodes'": {
			req:    newCSR(kubeletServerPEMOptions, pemOptions{org: "not-system:nodes"}),
			usages: kubeletServerUsages,
			exp:    false,
		},
		"does not default to kubelet-serving if CN does not have system:node: prefix": {
			req:    newCSR(kubeletServerPEMOptions, pemOptions{cn: "notprefixed"}),
			usages: kubeletServerUsages,
			exp:    false,
		},
		"does not default to kubelet-serving if it has an unexpected usage": {
			req:    newCSR(kubeletServerPEMOptions),
			usages: append(kubeletServerUsages, capi.UsageClientAuth),
			exp:    false,
		},
		"does not default to kubelet-serving if it is missing an expected usage": {
			req:    newCSR(kubeletServerPEMOptions),
			usages: kubeletServerUsages[1:],
			exp:    false,
		},
		"does not default to kubelet-serving if it does not specify any dnsNames or ipAddresses": {
			req:    newCSR(kubeletServerPEMOptions, pemOptions{ipAddresses: []net.IP{}, dnsNames: []string{}}),
			usages: kubeletServerUsages[1:],
			exp:    false,
		},
		"does not default to kubelet-serving if it specifies a URI SAN": {
			req:    newCSR(kubeletServerPEMOptions, pemOptions{uris: []string{"http://something"}}),
			usages: kubeletServerUsages,
			exp:    false,
		},
		"does not default to kubelet-serving if it specifies an emailAddress SAN": {
			req:    newCSR(kubeletServerPEMOptions, pemOptions{emailAddresses: []string{"something"}}),
			usages: kubeletServerUsages,
			exp:    false,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := IsKubeletServingCSR(test.req, test.usages)
			if test.exp != got {
				t.Errorf("unexpected IsKubeletClientCSR output: exp=%v, got=%v", test.exp, got)
			}
		})
	}
}

func TestIsKubeletClientCSR(t *testing.T) {
	newCSR := func(base pemOptions, overlays ...pemOptions) *x509.CertificateRequest {
		b := csrWithOpts(base, overlays...)
		csr, err := ParseCSR(b)
		if err != nil {
			t.Fatal(err)
		}
		return csr
	}
	tests := map[string]struct {
		req    *x509.CertificateRequest
		usages []capi.KeyUsage
		exp    bool
	}{
		"defaults for kube-apiserver-client-kubelet": {
			req:    newCSR(kubeletClientPEMOptions),
			usages: kubeletClientUsages,
			exp:    true,
		},
		"does not default to kube-apiserver-client-kubelet if org is not 'system:nodes'": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{org: "not-system:nodes"}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if a dnsName is set": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{dnsNames: []string{"something"}}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if an emailAddress is set": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{emailAddresses: []string{"something"}}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if a uri SAN is set": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{uris: []string{"http://something"}}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if an ipAddress is set": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{ipAddresses: []net.IP{{0, 0, 0, 0}}}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if CN does not have 'system:node:' prefix": {
			req:    newCSR(kubeletClientPEMOptions, pemOptions{cn: "not-prefixed"}),
			usages: kubeletClientUsages,
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if it has an unexpected usage": {
			req:    newCSR(kubeletClientPEMOptions),
			usages: append(kubeletClientUsages, capi.UsageServerAuth),
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if it is missing an expected usage": {
			req:    newCSR(kubeletClientPEMOptions),
			usages: kubeletClientUsages[1:],
			exp:    false,
		},
		"does not default to kube-apiserver-client-kubelet if it is missing an expected usage without key encipherment": {
			req:    newCSR(kubeletClientPEMOptions),
			usages: kubeletClientUsagesNoRSA[1:],
			exp:    false,
		},
		"default to kube-apiserver-client-kubelet without key encipherment": {
			req:    newCSR(kubeletClientPEMOptions),
			usages: kubeletClientUsagesNoRSA,
			exp:    true,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := IsKubeletClientCSR(test.req, test.usages)
			if test.exp != got {
				t.Errorf("unexpected IsKubeletClientCSR output: exp=%v, got=%v", test.exp, got)
			}
		})
	}
}

var (
	kubeletClientUsages = []capi.KeyUsage{
		capi.UsageDigitalSignature,
		capi.UsageKeyEncipherment,
		capi.UsageClientAuth,
	}
	kubeletClientUsagesNoRSA = []capi.KeyUsage{
		capi.UsageDigitalSignature,
		capi.UsageClientAuth,
	}
	kubeletClientPEMOptions = pemOptions{
		cn:  "system:node:nodename",
		org: "system:nodes",
	}

	kubeletServerUsages = []capi.KeyUsage{
		capi.UsageDigitalSignature,
		capi.UsageKeyEncipherment,
		capi.UsageServerAuth,
	}
	kubeletServerUsagesNoRSA = []capi.KeyUsage{
		capi.UsageDigitalSignature,
		capi.UsageServerAuth,
	}
	kubeletServerPEMOptions = pemOptions{
		cn:          "system:node:requester-name",
		org:         "system:nodes",
		dnsNames:    []string{"node-server-name"},
		ipAddresses: []net.IP{{0, 0, 0, 0}},
	}
)

func TestSetDefaults_CertificateSigningRequestSpec(t *testing.T) {
	strPtr := func(s string) *string { return &s }
	tests := map[string]struct {
		csr                capi.CertificateSigningRequestSpec
		expectedSignerName string
		expectedUsages     []capi.KeyUsage
	}{
		"defaults to legacy-unknown if request is not a CSR": {
			csr: capi.CertificateSigningRequestSpec{
				Request: []byte("invalid data"),
				Usages:  kubeletServerUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default signerName if signerName is already set": {
			csr: capi.CertificateSigningRequestSpec{
				Request:    csrWithOpts(kubeletServerPEMOptions),
				Usages:     kubeletServerUsages,
				SignerName: strPtr("example.com/not-kubelet-serving"),
			},
			expectedSignerName: "example.com/not-kubelet-serving",
		},
		"defaults usages if not set": {
			csr: capi.CertificateSigningRequestSpec{
				Request:    csrWithOpts(kubeletServerPEMOptions),
				SignerName: strPtr("example.com/test"),
			},
			expectedSignerName: "example.com/test",
			expectedUsages:     []capi.KeyUsage{capi.UsageDigitalSignature, capi.UsageKeyEncipherment},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// create a deepcopy to be sure we don't modify anything in-place
			csrSpec := test.csr.DeepCopy()
			SetDefaults_CertificateSigningRequestSpec(csrSpec)
			if *csrSpec.SignerName != test.expectedSignerName {
				t.Errorf("expected signerName to be defaulted to %q but it is %q", test.expectedSignerName, *csrSpec.SignerName)
			}

			// only check expectedUsages if it is non-nil
			if test.expectedUsages != nil {
				if !reflect.DeepEqual(test.expectedUsages, csrSpec.Usages) {
					t.Errorf("expected usages to be defaulted to %v but it is %v", test.expectedUsages, csrSpec.Usages)
				}
			}
		})
	}
}

func TestSetDefaults_CertificateSigningRequestSpec_KubeletServing(t *testing.T) {
	tests := map[string]struct {
		csr                capi.CertificateSigningRequestSpec
		expectedSignerName string
	}{
		"defaults for kubelet-serving": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions),
				Usages:   kubeletServerUsages,
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.KubeletServingSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if org is not 'system:nodes'": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions, pemOptions{org: "not-system:nodes"}),
				Usages:   kubeletServerUsages,
				Username: "system:node:not-requester-name",
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if CN does not have system:node: prefix": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions, pemOptions{cn: "notprefixed"}),
				Usages:   kubeletServerUsages,
				Username: "notprefixed",
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if it has an unexpected usage": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions),
				Usages:   append(kubeletServerUsages, capi.UsageClientAuth),
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if it is missing an expected usage": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletServerPEMOptions),
				// Remove the first usage in 'kubeletServerUsages'
				Usages:   kubeletServerUsages[1:],
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if it does not specify any dnsNames or ipAddresses": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions, pemOptions{ipAddresses: []net.IP{}, dnsNames: []string{}}),
				Usages:   kubeletServerUsages,
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if it specifies a URI SAN": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions, pemOptions{uris: []string{"http://something"}}),
				Usages:   kubeletServerUsages,
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kubelet-serving if it specifies an emailAddress SAN": {
			csr: capi.CertificateSigningRequestSpec{
				Request:  csrWithOpts(kubeletServerPEMOptions, pemOptions{emailAddresses: []string{"something"}}),
				Usages:   kubeletServerUsages,
				Username: kubeletServerPEMOptions.cn,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// create a deepcopy to be sure we don't modify anything in-place
			csrSpec := test.csr.DeepCopy()
			SetDefaults_CertificateSigningRequestSpec(csrSpec)
			if *csrSpec.SignerName != test.expectedSignerName {
				t.Errorf("expected signerName to be defaulted to %q but it is %q", test.expectedSignerName, *csrSpec.SignerName)
			}
		})
	}
}

func TestSetDefaults_CertificateSigningRequestSpec_KubeletClient(t *testing.T) {
	tests := map[string]struct {
		csr                capi.CertificateSigningRequestSpec
		expectedSignerName string
	}{
		"defaults for kube-apiserver-client-kubelet": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.KubeAPIServerClientKubeletSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if org is not 'system:nodes'": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{org: "not-system:nodes"}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if a dnsName is set": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{dnsNames: []string{"something"}}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if an emailAddress is set": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{emailAddresses: []string{"something"}}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if a uri SAN is set": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{uris: []string{"http://something"}}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if an ipAddress is set": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{ipAddresses: []net.IP{{0, 0, 0, 0}}}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if CN does not have 'system:node:' prefix": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions, pemOptions{cn: "not-prefixed"}),
				Usages:  kubeletClientUsages,
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if it has an unexpected usage": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions),
				Usages:  append(kubeletClientUsages, capi.UsageServerAuth),
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
		"does not default to kube-apiserver-client-kubelet if it is missing an expected usage": {
			csr: capi.CertificateSigningRequestSpec{
				Request: csrWithOpts(kubeletClientPEMOptions),
				// Remove the first usage in 'kubeletClientUsages'
				Usages: kubeletClientUsages[1:],
			},
			expectedSignerName: capi.LegacyUnknownSignerName,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// create a deepcopy to be sure we don't modify anything in-place
			csrSpec := test.csr.DeepCopy()
			SetDefaults_CertificateSigningRequestSpec(csrSpec)
			if *csrSpec.SignerName != test.expectedSignerName {
				t.Errorf("expected signerName to be defaulted to %q but it is %q", test.expectedSignerName, *csrSpec.SignerName)
			}
		})
	}
}

type pemOptions struct {
	cn             string
	org            string
	ipAddresses    []net.IP
	dnsNames       []string
	emailAddresses []string
	uris           []string
}

// overlayPEMOptions overlays one set of pemOptions on top of another to allow
// for easily overriding a single field in the options
func overlayPEMOptions(opts ...pemOptions) pemOptions {
	if len(opts) == 0 {
		return pemOptions{}
	}
	base := opts[0]
	for _, opt := range opts[1:] {
		if opt.cn != "" {
			base.cn = opt.cn
		}
		if opt.org != "" {
			base.org = opt.org
		}
		if opt.ipAddresses != nil {
			base.ipAddresses = opt.ipAddresses
		}
		if opt.dnsNames != nil {
			base.dnsNames = opt.dnsNames
		}
		if opt.emailAddresses != nil {
			base.emailAddresses = opt.emailAddresses
		}
		if opt.uris != nil {
			base.uris = opt.uris
		}
	}
	return base
}

func csrWithOpts(base pemOptions, overlays ...pemOptions) []byte {
	opts := overlayPEMOptions(append([]pemOptions{base}, overlays...)...)
	uris := make([]*url.URL, len(opts.uris))
	for i, s := range opts.uris {
		u, err := url.ParseRequestURI(s)
		if err != nil {
			panic(err)
		}
		uris[i] = u
	}
	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   opts.cn,
			Organization: []string{opts.org},
		},
		IPAddresses:    opts.ipAddresses,
		DNSNames:       opts.dnsNames,
		EmailAddresses: opts.emailAddresses,
		URIs:           uris,
	}

	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		panic(err)
	}

	csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, key)
	if err != nil {
		panic(err)
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csrDER,
	}

	p := pem.EncodeToMemory(csrPemBlock)
	if p == nil {
		panic("invalid pem block")
	}

	return p
}
