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

package storage

import (
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/utils/clock"
	testclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

func newStorage(t *testing.T, authz authorizer.Authorizer, clock clock.PassiveClock) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, certificates.SchemeGroupVersion.WithResource("podcertificaterequests").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "podcertificaterequests",
	}
	storage, statusStorage, err := NewREST(restOptions, authz, clock)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, statusStorage, server
}

func TestCreate(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}

	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))
	_, _, ed25519PubPKIX2, ed25519Proof2 := mustMakeEd25519KeyAndProof(t, []byte("other-value"))

	test.TestCreate(
		// Valid PCR
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:                "k8s.io/foo",
				PodName:                   "pod-1",
				PodUID:                    podUID1,
				ServiceAccountName:        "sa-1",
				ServiceAccountUID:         types.UID("sa-uid-1"),
				NodeName:                  "node-1",
				NodeUID:                   types.UID("node-uid-1"),
				MaxExpirationSeconds:      ptr.To[int32](86400),
				PKIXPublicKey:             ed25519PubPKIX1,
				ProofOfPossession:         ed25519Proof1,
				UnverifiedUserAnnotations: map[string]string{"test/foo": "bar"},
			},
		},
		// Invalid PCR -- proof-of-possession signed wrong value
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:                "k8s.io/foo",
				PodName:                   "pod-1",
				PodUID:                    podUID1,
				ServiceAccountName:        "sa-1",
				ServiceAccountUID:         types.UID("sa-uid-1"),
				NodeName:                  "node-1",
				NodeUID:                   types.UID("node-uid-1"),
				MaxExpirationSeconds:      ptr.To[int32](86400),
				PKIXPublicKey:             ed25519PubPKIX2,
				ProofOfPossession:         ed25519Proof2,
				UnverifiedUserAnnotations: map[string]string{"test/foo": "bar"},
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestUpdate(
		// Valid PCR as a base
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:                "k8s.io/foo",
				PodName:                   "pod-1",
				PodUID:                    podUID1,
				ServiceAccountName:        "sa-1",
				ServiceAccountUID:         types.UID("sa-uid-1"),
				NodeName:                  "node-1",
				NodeUID:                   types.UID("node-uid-1"),
				MaxExpirationSeconds:      ptr.To[int32](86400),
				PKIXPublicKey:             ed25519PubPKIX1,
				ProofOfPossession:         ed25519Proof1,
				UnverifiedUserAnnotations: map[string]string{"test/foo": "bar"},
			},
		},
		// Valid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			if pcr.ObjectMeta.Annotations == nil {
				pcr.ObjectMeta.Annotations = map[string]string{}
			}
			pcr.ObjectMeta.Annotations["k8s.io/cool-annotation"] = "value"
			return pcr
		},
		// Invalid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			pcr.Spec.SignerName = "test.k8s.io/new-signer"
			return pcr
		},
	)
}

func TestUpdateStompsStatus(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestUpdate(
		// Valid PCR as a base
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:                "k8s.io/foo",
				PodName:                   "pod-1",
				PodUID:                    podUID1,
				ServiceAccountName:        "sa-1",
				ServiceAccountUID:         types.UID("sa-uid-1"),
				NodeName:                  "node-1",
				NodeUID:                   types.UID("node-uid-1"),
				MaxExpirationSeconds:      ptr.To[int32](86400),
				PKIXPublicKey:             ed25519PubPKIX1,
				ProofOfPossession:         ed25519Proof1,
				UnverifiedUserAnnotations: map[string]string{"test/foo": "bar"},
			},
		},
		// Valid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			// The strategy should stomp status.
			pcr.Status.NotAfter = ptr.To(metav1.NewTime(mustParseTime(t, "2025-01-01T00:00:00Z")))
			return pcr
		},
	)

}

func TestUpdateStatus(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	caCertDER, caPrivKey := mustMakeCA(t)
	podUID1 := types.UID("pod-uid-1")
	_, ed25519Pub1, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))
	pod1Cert1 := mustSignCertForPublicKey(t, 24*time.Hour, ed25519Pub1, caCertDER, caPrivKey)

	test.TestUpdate(
		// Valid PCR as a base
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
		// Valid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			pcr.Status = certificates.PodCertificateRequestStatus{
				Conditions: []metav1.Condition{
					{
						Type:               certificates.PodCertificateRequestConditionTypeIssued,
						Status:             metav1.ConditionTrue,
						Reason:             "Whatever",
						Message:            "Foo message",
						LastTransitionTime: metav1.NewTime(time.Now()),
					},
				},
				CertificateChain: pod1Cert1,
				NotBefore:        ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-01T00:00:00Z"))),
				BeginRefreshAt:   ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-01T12:00:00Z"))),
				NotAfter:         ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-02T00:00:00Z"))),
			}
			return pcr
		},
		// Invalid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			pcr.Spec.SignerName = "test.k8s.io/new-signer"
			return pcr
		},
	)
}

func TestUpdateStatusStompsSpec(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestUpdate(
		// Valid PCR as a base
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
		// Valid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			// The stategy should stomp spec.
			pcr.Spec.SignerName = "foo.com/bar"
			return pcr
		},
	)
}

func TestUpdateStatusFailsWhenAuthorizerDenies(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionNoOpinion,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	caCertDER, caPrivKey := mustMakeCA(t)
	podUID1 := types.UID("pod-uid-1")
	_, ed25519Pub1, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))
	pod1Cert1 := mustSignCertForPublicKey(t, 24*time.Hour, ed25519Pub1, caCertDER, caPrivKey)

	test.TestUpdate(
		// Valid PCR as a base
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
		// Valid update function
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			return pcr
		},
		// Invalid update function -- normally a valid update, but above we have
		// configured the authorizer to never return DecisionAllow.
		func(object runtime.Object) runtime.Object {
			pcr := object.(*certificates.PodCertificateRequest)
			pcr.Status = certificates.PodCertificateRequestStatus{
				Conditions: []metav1.Condition{
					{
						Type:               certificates.PodCertificateRequestConditionTypeIssued,
						Status:             metav1.ConditionTrue,
						Reason:             "Whatever",
						Message:            "Foo message",
						LastTransitionTime: metav1.NewTime(time.Now()),
					},
				},
				CertificateChain: pod1Cert1,
				NotBefore:        ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-01T00:00:00Z"))),
				BeginRefreshAt:   ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-01T12:00:00Z"))),
				NotAfter:         ptr.To(metav1.NewTime(mustParseTime(t, "1970-01-02T00:00:00Z"))),
			}
			return pcr
		},
	)

}

func TestDelete(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestDelete(
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
	)
}

func TestGet(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestGet(
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
	)
}

func TestList(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestList(
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
	)
}

func TestWatch(t *testing.T) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")))
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")
	_, _, ed25519PubPKIX1, ed25519Proof1 := mustMakeEd25519KeyAndProof(t, []byte(podUID1))

	test.TestWatch(
		&certificates.PodCertificateRequest{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: test.TestNamespace(),
				Name:      "foo",
			},
			Spec: certificates.PodCertificateRequestSpec{
				SignerName:           "k8s.io/foo",
				PodName:              "pod-1",
				PodUID:               podUID1,
				ServiceAccountName:   "sa-1",
				ServiceAccountUID:    types.UID("sa-uid-1"),
				NodeName:             "node-1",
				NodeUID:              types.UID("node-uid-1"),
				MaxExpirationSeconds: ptr.To[int32](86400),
				PKIXPublicKey:        ed25519PubPKIX1,
				ProofOfPossession:    ed25519Proof1,
			},
		},
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{
				"metadata.namespace": test.TestNamespace(),
				"metadata.name":      "foo",
				"spec.signerName":    "k8s.io/foo",
				"spec.nodeName":      "node-1",
			},
		},
		// not matching fields
		[]fields.Set{
			{
				"metadata.namespace": test.TestNamespace(),
				"metadata.name":      "foo",
				"spec.signerName":    "k8s.io/othersigner",
				"spec.nodeName":      "node-1",
			},
		},
	)
}

func mustMakeCA(t *testing.T) ([]byte, ed25519.PrivateKey) {
	signPub, signPriv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating CA signing key: %v", err)
	}

	caCertTemplate := &x509.Certificate{
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		NotBefore:             mustParseTime(t, "1970-01-01T00:00:00Z"),
		NotAfter:              mustParseTime(t, "1971-01-01T00:00:00Z"),
	}

	caCertDER, err := x509.CreateCertificate(rand.Reader, caCertTemplate, caCertTemplate, signPub, signPriv)
	if err != nil {
		t.Fatalf("Error while creating CA certificate: %v", err)
	}

	return caCertDER, signPriv
}

func mustParseTime(t *testing.T, stamp string) time.Time {
	got, err := time.Parse(time.RFC3339, stamp)
	if err != nil {
		t.Fatalf("Error while parsing timestamp: %v", err)
	}
	return got
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

func mustSignCertForPublicKey(t *testing.T, validity time.Duration, subjectPublicKey crypto.PublicKey, caCertDER []byte, caPrivateKey crypto.PrivateKey) string {
	certTemplate := &x509.Certificate{
		Subject: pkix.Name{
			CommonName: "foo",
		},
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		NotBefore:   mustParseTime(t, "1970-01-01T00:00:00Z"),
		NotAfter:    mustParseTime(t, "1970-01-01T00:00:00Z").Add(validity),
	}

	caCert, err := x509.ParseCertificate(caCertDER)
	if err != nil {
		t.Fatalf("Error while parsing CA certificate: %v", err)
	}

	certDER, err := x509.CreateCertificate(rand.Reader, certTemplate, caCert, subjectPublicKey, caPrivateKey)
	if err != nil {
		t.Fatalf("Error while signing subject certificate: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDER,
	})

	return string(certPEM)
}
