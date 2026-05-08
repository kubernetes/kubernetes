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

package clustertrustbundle

import (
	"crypto/ed25519"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	mathrand "math/rand"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

var apiVersions = []string{"v1alpha1", "v1beta1"}

func mustMakeTestCert(t *testing.T) string {
	t.Helper()
	gen := mathrand.New(mathrand.NewSource(12345))

	pub, priv, err := ed25519.GenerateKey(gen)
	if err != nil {
		t.Fatalf("Error while generating key: %v", err)
	}

	cert, err := x509.CreateCertificate(gen, &x509.Certificate{
		SerialNumber:          big.NewInt(0),
		Subject:               pkix.Name{CommonName: "root1"},
		IsCA:                  true,
		BasicConstraintsValid: true,
	}, &x509.Certificate{
		SerialNumber:          big.NewInt(0),
		Subject:               pkix.Name{CommonName: "root1"},
		IsCA:                  true,
		BasicConstraintsValid: true,
	}, pub, priv)
	if err != nil {
		t.Fatalf("Error while making certificate: %v", err)
	}

	return string(pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert,
	}))
}

func mkValidBundle(t *testing.T, tweaks ...func(*certificates.ClusterTrustBundle)) certificates.ClusterTrustBundle {
	t.Helper()
	bundle := certificates.ClusterTrustBundle{
		ObjectMeta: metav1.ObjectMeta{Name: "example.com:foo:bar"},
		Spec: certificates.ClusterTrustBundleSpec{
			SignerName:  "example.com/foo",
			TrustBundle: mustMakeTestCert(t),
		},
	}
	for _, tweak := range tweaks {
		tweak(&bundle)
	}
	return bundle
}

func tweakSignerName(signerName string) func(*certificates.ClusterTrustBundle) {
	return func(bundle *certificates.ClusterTrustBundle) {
		bundle.Spec.SignerName = signerName
	}
}

func tweakName(name string) func(*certificates.ClusterTrustBundle) {
	return func(bundle *certificates.ClusterTrustBundle) {
		bundle.ObjectMeta.Name = name
	}
}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateForDeclarative(t, apiVersion)
		})
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "certificates.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "clustertrustbundles",
		Name:              "test-bundle",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        certificates.ClusterTrustBundle
		expectedErrs field.ErrorList
	}{
		"valid bundle without signer": {
			input:        mkValidBundle(t, tweakName("test-bundle"), tweakSignerName("")),
			expectedErrs: field.ErrorList{},
		},
		"valid bundle with signer": {
			input:        mkValidBundle(t),
			expectedErrs: field.ErrorList{},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdateForDeclarative(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       certificates.ClusterTrustBundle
		updateObj    certificates.ClusterTrustBundle
		expectedErrs field.ErrorList
	}{
		"signerName unchanged - valid": {
			oldObj:    mkValidBundle(t),
			updateObj: mkValidBundle(t),
		},
		"signerName changed - invalid": {
			oldObj:    mkValidBundle(t),
			updateObj: mkValidBundle(t, tweakSignerName("example.com/bar")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "example.com:foo:bar", "ClusterTrustBundle for signerName example.com/bar must be named with prefix example.com:bar:").MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "signerName"), "example.com/bar", "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"signerName set from unset - invalid": {
			oldObj:    mkValidBundle(t, tweakName("no-signer-bundle"), tweakSignerName("")),
			updateObj: mkValidBundle(t, tweakName("no-signer-bundle")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "no-signer-bundle", "ClusterTrustBundle for signerName example.com/foo must be named with prefix example.com:foo:").MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "signerName"), "example.com/foo", "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"signerName unset from set - invalid": {
			oldObj:    mkValidBundle(t),
			updateObj: mkValidBundle(t, tweakSignerName("")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "example.com:foo:bar", `ClusterTrustBundle without signer name must not have ":" in its name`).MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "signerName"), "", "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "clustertrustbundles",
				Name:              tc.oldObj.Name,
				IsResourceRequest: true,
				Verb:              "update",
			})
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy, tc.expectedErrs)
		})
	}
}
