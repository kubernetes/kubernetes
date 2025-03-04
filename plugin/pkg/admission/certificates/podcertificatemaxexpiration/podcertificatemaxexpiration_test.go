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

package podcertificatemaxexpiration

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	certs "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/utils/ptr"
)

func TestAdmitOverwrite(t *testing.T) {
	handler := admissiontesting.WithReinvocationTesting(t, &Plugin{
		inspectedFeatureGates: true,
		enabled:               true,
	})

	testCases := []struct {
		desc    string
		pcr     *certs.PodCertificateRequest
		wantPCR *certs.PodCertificateRequest
	}{
		{
			desc: "Overwrite nil spec.maxExpirationSeconds",
			pcr: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName: "example.com/foo",
				},
			},
			wantPCR: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName:           "example.com/foo",
					MaxExpirationSeconds: ptr.To[int32](86400),
				},
			},
		},
		{
			desc: "Overwrite too-long spec.maxExpirationSeconds",
			pcr: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName:           "example.com/foo",
					MaxExpirationSeconds: ptr.To[int32](3 * 86400),
				},
			},
			wantPCR: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName:           "example.com/foo",
					MaxExpirationSeconds: ptr.To[int32](86400),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := handler.Admit(
				context.TODO(),
				admission.NewAttributesRecord(
					tc.pcr,
					nil,
					certs.Kind("PodCertificateRequest").WithVersion("v1alpha1"),
					tc.pcr.ObjectMeta.Namespace,
					tc.pcr.ObjectMeta.Name,
					certs.Resource("podcertificaterequests").WithVersion("v1alpha1"),
					"",
					admission.Create,
					&metav1.CreateOptions{},
					false,
					nil,
				),
				nil,
			)
			if err != nil {
				t.Errorf("Unexpected error returned from admission handler")
			}

			if diff := cmp.Diff(tc.pcr, tc.wantPCR); diff != "" {
				t.Errorf("Bad result from admission handler; diff (-got +want)\n%s", diff)
			}
		})
	}
}

func TestPassValidation(t *testing.T) {
	handler := &Plugin{
		inspectedFeatureGates: true,
		enabled:               true,
	}

	testCases := []struct {
		desc string
		pcr  *certs.PodCertificateRequest
	}{
		{
			desc: "expiration <= 86400 ",
			pcr: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName:           "example.com/foo",
					MaxExpirationSeconds: ptr.To[int32](86400),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := handler.Validate(
				context.TODO(),
				admission.NewAttributesRecord(
					tc.pcr,
					nil,
					certs.Kind("PodCertificateRequest").WithVersion("v1alpha1"),
					tc.pcr.ObjectMeta.Namespace,
					tc.pcr.ObjectMeta.Name,
					certs.Resource("podcertificaterequests").WithVersion("v1alpha1"),
					"",
					admission.Create,
					&metav1.CreateOptions{},
					false,
					nil,
				),
				nil,
			)
			if err != nil {
				t.Errorf("Unexpected error returned from admission handler: %v", err)
			}
		})
	}
}

func TestFailValidation(t *testing.T) {
	handler := &Plugin{
		inspectedFeatureGates: true,
		enabled:               true,
	}

	testCases := []struct {
		desc string
		pcr  *certs.PodCertificateRequest
	}{
		{
			desc: "Missing expiration rejected ",
			pcr: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName: "example.com/foo",
				},
			},
		},
		{
			desc: "Expiration > 86400 rejected",
			pcr: &certs.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "foo",
				},
				Spec: certs.PodCertificateRequestSpec{
					SignerName:           "example.com/foo",
					MaxExpirationSeconds: ptr.To[int32](3 * 86400),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := handler.Validate(
				context.TODO(),
				admission.NewAttributesRecord(
					tc.pcr,
					nil,
					certs.Kind("PodCertificateRequest").WithVersion("v1alpha1"),
					tc.pcr.ObjectMeta.Namespace,
					tc.pcr.ObjectMeta.Name,
					certs.Resource("podcertificaterequests").WithVersion("v1alpha1"),
					"",
					admission.Create,
					&metav1.CreateOptions{},
					false,
					nil,
				),
				nil,
			)
			if err == nil {
				t.Errorf("Wanted error from admission handler, but got nil")
			}
		})
	}
}
