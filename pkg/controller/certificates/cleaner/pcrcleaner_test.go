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

package cleaner

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"testing"
	"time"

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	testclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestPCRCleaner(t *testing.T) {
	now := mustParseRFC3339(t, "2025-01-01T00:30:00Z")
	clock := testclock.NewFakePassiveClock(now)

	podUID1 := "pod-1-uid"
	_, _, pubPKIX1, proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	testCases := []struct {
		desc              string
		pcr               *certsv1beta1.PodCertificateRequest
		wantErrRecognizer func(error) bool
	}{
		{
			desc: "Pending request within the threshold should be left alone",
			pcr: &certsv1beta1.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "foo",
					Name:              "bar",
					CreationTimestamp: metav1.NewTime(mustParseRFC3339(t, "2025-01-01T00:15:01Z")),
				},
				Spec: certsv1beta1.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID(podUID1),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
			},
			wantErrRecognizer: errorIsNil,
		},
		{
			desc: "Pending request outside the threshold should be deleted",
			pcr: &certsv1beta1.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "foo",
					Name:              "bar",
					CreationTimestamp: metav1.NewTime(mustParseRFC3339(t, "2025-01-01T00:14:59Z")),
				},
				Spec: certsv1beta1.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID(podUID1),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
			},
			wantErrRecognizer: k8serrors.IsNotFound,
		},
		{
			desc: "Terminal request within the threshold should be left alone",
			pcr: &certsv1beta1.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "foo",
					Name:              "bar",
					CreationTimestamp: metav1.NewTime(mustParseRFC3339(t, "2025-01-01T00:15:01Z")),
				},
				Spec: certsv1beta1.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID(podUID1),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
				Status: certsv1beta1.PodCertificateRequestStatus{
					Conditions: []metav1.Condition{
						{
							Type:    certsv1beta1.PodCertificateRequestConditionTypeDenied,
							Status:  metav1.ConditionTrue,
							Reason:  "Foo",
							Message: "abc",
						},
					},
				},
			},
			wantErrRecognizer: errorIsNil,
		},
		{
			desc: "Terminal request outside the threshold should be deleted",
			pcr: &certsv1beta1.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "foo",
					Name:              "bar",
					CreationTimestamp: metav1.NewTime(mustParseRFC3339(t, "2025-01-01T00:14:59Z")),
				},
				Spec: certsv1beta1.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID(podUID1),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
				Status: certsv1beta1.PodCertificateRequestStatus{
					Conditions: []metav1.Condition{
						{
							Type:    certsv1beta1.PodCertificateRequestConditionTypeDenied,
							Status:  metav1.ConditionTrue,
							Reason:  "Foo",
							Message: "abc",
						},
					},
				},
			},
			wantErrRecognizer: k8serrors.IsNotFound,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()
			kc := fake.NewClientset(tc.pcr)
			cleaner := &PCRCleanerController{
				client:          kc,
				clock:           clock,
				threshold:       15 * time.Minute,
				pollingInterval: 1 * time.Minute,
			}

			// Simulate a pass of the cleaner worker by listing all PCRs and
			// calling handle() on them.

			pcrList, err := kc.CertificatesV1beta1().PodCertificateRequests(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("Unexpected error listing PCRs: %v", err)
			}
			for _, pcr := range pcrList.Items {
				if err := cleaner.handle(ctx, &pcr); err != nil {
					t.Fatalf("Unexpected error calling handle(): %v", err)
				}
			}

			// Now check on the test case's PCR, to see if it was deleted or not
			// according to our expectation.
			_, err = kc.CertificatesV1beta1().PodCertificateRequests(tc.pcr.ObjectMeta.Namespace).Get(ctx, tc.pcr.ObjectMeta.Name, metav1.GetOptions{})
			if !tc.wantErrRecognizer(err) {
				t.Errorf("Bad error output: %v", err)
			}
		})
	}

}

func errorIsNil(err error) bool {
	return err == nil
}

func mustParseRFC3339(t *testing.T, stamp string) time.Time {
	parsed, err := time.Parse(time.RFC3339, stamp)
	if err != nil {
		t.Fatalf("Unexpected error parsing time: %v", err)
	}
	return parsed
}

func mustMakeEd25519KeyAndProof(t *testing.T, toBeSigned []byte) (ed25519.PrivateKey, ed25519.PublicKey, []byte, []byte) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating ed25519 key: %v", err)
	}
	pubPKIX, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		t.Fatalf("Error while marshaling PKIX public key: %v", err)
	}
	sig := ed25519.Sign(priv, toBeSigned)
	return priv, pub, pubPKIX, sig
}
