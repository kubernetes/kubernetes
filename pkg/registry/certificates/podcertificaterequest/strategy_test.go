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
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

func TestWarningsOnCreate(t *testing.T) {
	strategy := NewStrategy()

	var wantWarnings []string
	gotWarnings := strategy.WarningsOnCreate(context.Background(), &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowCreateOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowCreateOnUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	var wantWarnings []string
	gotWarnings := strategy.WarningsOnUpdate(context.Background(), &certificates.PodCertificateRequest{}, &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowUnconditionalUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowUnconditionalUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestPrepareForCreate(t *testing.T) {
	// PrepareForCreate should stomp any existing status fields.

	strategy := NewStrategy()

	_, _, pubPKIX1, proof1 := mustMakeEd25519KeyAndProof(t, []byte("pod-1-uid"))

	processedPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Denied",
					Status:  metav1.ConditionFalse,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	wantPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
	}

	strategy.PrepareForCreate(context.Background(), processedPCR)

	if diff := cmp.Diff(processedPCR, wantPCR); diff != "" {
		t.Errorf("Bad processed PCR; diff (-got +want)\n%s", diff)
	}
}

func TestPrepareForUpdate(t *testing.T) {
	// PrepareForUpdate should stomp any existing status fields.

	strategy := NewStrategy()

	_, _, pubPKIX1, proof1 := mustMakeEd25519KeyAndProof(t, []byte("pod-1-uid"))

	oldPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Denied",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	processedPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Failed",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	wantPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Denied",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	strategy.PrepareForUpdate(context.Background(), processedPCR, oldPCR)

	if diff := cmp.Diff(processedPCR, wantPCR); diff != "" {
		t.Errorf("Bad processed PCR; diff (-got +want)\n%s", diff)
	}
}

func TestStatusPrepareForUpdate(t *testing.T) {
	// StatusStrategy.PrepareForUpdate should reset all spec fields and most
	// metadata fields in the new object.

	strategy := NewStrategy()
	authz := &FakeAuthorizer{
		authorized: authorizer.DecisionAllow,
	}
	statusStrategy := NewStatusStrategy(strategy, authz, clock.RealClock{})

	_, _, pubPKIX1, proof1 := mustMakeEd25519KeyAndProof(t, []byte("pod-1-uid"))

	oldPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Denied",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	processedPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/different-value",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Failed",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	wantPCR := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "foo",
			Name:      "bar",
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:           "foo.com/abc",
			PodName:              "pod-1",
			PodUID:               types.UID("pod-1-uid"),
			ServiceAccountName:   "sa-1",
			ServiceAccountUID:    "sa-uid-1",
			NodeName:             "node-1",
			NodeUID:              "node-uid-1",
			MaxExpirationSeconds: ptr.To[int32](86400),
			PKIXPublicKey:        pubPKIX1,
			ProofOfPossession:    proof1,
		},
		Status: certificates.PodCertificateRequestStatus{
			Conditions: []metav1.Condition{
				{
					Type:    "Failed",
					Status:  metav1.ConditionTrue,
					Reason:  "Foo",
					Message: "Foo message",
				},
			},
		},
	}

	statusStrategy.PrepareForUpdate(context.Background(), processedPCR, oldPCR)

	if diff := cmp.Diff(processedPCR, wantPCR); diff != "" {
		t.Errorf("Bad processed PCR; diff (-got +want)\n%s", diff)
	}
}

func TestStatusValidateUpdate(t *testing.T) {

	_, _, pubPKIX1, proof1 := mustMakeEd25519KeyAndProof(t, []byte("pod-1-uid"))

	testCases := []struct {
		desc                 string
		oldPCR, newPCR       *certificates.PodCertificateRequest
		authz                authorizer.Authorizer
		wantValidationErrors field.ErrorList
	}{
		{
			desc: "No errors when the caller is authorized",
			oldPCR: &certificates.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       "foo",
					Name:            "bar",
					ResourceVersion: "1",
				},
				Spec: certificates.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID("pod-1-uid"),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
			},
			newPCR: &certificates.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       "foo",
					Name:            "bar",
					ResourceVersion: "1",
				},
				Spec: certificates.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID("pod-1-uid"),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
				Status: certificates.PodCertificateRequestStatus{
					Conditions: []metav1.Condition{
						{
							Type:               "Failed",
							Status:             metav1.ConditionTrue,
							Reason:             "Foo",
							Message:            "Foo message",
							LastTransitionTime: metav1.NewTime(time.Now()),
						},
					},
				},
			},
			authz: &FakeAuthorizer{
				authorized: authorizer.DecisionAllow,
			},
		},
		{
			desc: "Error when the caller is not authorized",
			oldPCR: &certificates.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       "foo",
					Name:            "bar",
					ResourceVersion: "1",
				},
				Spec: certificates.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID("pod-1-uid"),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
			},
			newPCR: &certificates.PodCertificateRequest{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       "foo",
					Name:            "bar",
					ResourceVersion: "1",
				},
				Spec: certificates.PodCertificateRequestSpec{
					SignerName:           "foo.com/abc",
					PodName:              "pod-1",
					PodUID:               types.UID("pod-1-uid"),
					ServiceAccountName:   "sa-1",
					ServiceAccountUID:    "sa-uid-1",
					NodeName:             "node-1",
					NodeUID:              "node-uid-1",
					MaxExpirationSeconds: ptr.To[int32](86400),
					PKIXPublicKey:        pubPKIX1,
					ProofOfPossession:    proof1,
				},
				Status: certificates.PodCertificateRequestStatus{
					Conditions: []metav1.Condition{
						{
							Type:               "Failed",
							Status:             metav1.ConditionTrue,
							Reason:             "Foo",
							Message:            "Foo message",
							LastTransitionTime: metav1.NewTime(time.Now()),
						},
					},
				},
			},
			authz: &FakeAuthorizer{
				authorized: authorizer.DecisionNoOpinion,
				reason:     "not authorized",
			},
			wantValidationErrors: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "signerName"), "User \"bob\" is not permitted to \"sign\" for signer \"foo.com/abc\""),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := context.Background()
			ctx = request.WithUser(ctx, &user.DefaultInfo{Name: "bob"})

			strategy := NewStrategy()
			statusStrategy := NewStatusStrategy(strategy, tc.authz, clock.RealClock{})

			gotValidationErrors := statusStrategy.ValidateUpdate(ctx, tc.newPCR, tc.oldPCR)
			if diff := cmp.Diff(gotValidationErrors, tc.wantValidationErrors); diff != "" {
				t.Errorf("Wrong validation errors; diff (-got +want)\n%s", diff)
			}
		})
	}

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

type FakeAuthorizer struct {
	authorized authorizer.Decision
	reason     string
	err        error
}

func (f *FakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return f.authorized, f.reason, f.err
}
