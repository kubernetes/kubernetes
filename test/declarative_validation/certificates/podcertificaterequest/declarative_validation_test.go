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

package podcertificaterequest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	certificates "k8s.io/kubernetes/pkg/apis/certificates"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	registry "k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"testing"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1beta1"}

// Helper function to create a baseline valid PodCertificateRequest with optional tweaks
func mkPodCertificateRequest(tweaks ...func(*certificates.PodCertificateRequest)) certificates.PodCertificateRequest {
	obj := func() certificates.PodCertificateRequest {
		expiration := int32(3600)
		_, priv, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			panic(err)
		}
		template := &x509.CertificateRequest{}
		csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, priv)
		if err != nil {
			panic(err)
		}
		return certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Name: "valid-resource-name",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "kubernetes.io/kube-apiserver-client-pod",
				PodName:              "valid-pod-name",
				PodUID:               types.UID("a0123456-7890-abcd-ef01-234567890abc"),
				ServiceAccountName:   "default",
				ServiceAccountUID:    types.UID("b0123456-7890-abcd-ef01-234567890abc"),
				NodeName:             types.NodeName("valid-node-name"),
				NodeUID:              types.UID("c0123456-7890-abcd-ef01-234567890abc"),
				MaxExpirationSeconds: &expiration,
				StubPKCS10Request:    csrDER,
			},
		}
	}()
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy()
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "podcertificaterequests",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkPodCertificateRequest(func(o *certificates.PodCertificateRequest) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy()
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "certificates.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "podcertificaterequests",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkPodCertificateRequest(func(o *certificates.PodCertificateRequest) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)
		})
	}
}
