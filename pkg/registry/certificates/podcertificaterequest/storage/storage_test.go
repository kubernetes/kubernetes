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
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	basecompatibility "k8s.io/component-base/compatibility"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/utils/clock"
	testclock "k8s.io/utils/clock/testing"
)

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

// newStorage creates the REST storages. Unless usePKCS10 is set, the storage
// version is v1beta1: v1 has no PKIXPublicKey/ProofOfPossession, so those can
// only be stored with v1beta1 (v1 was introduced in 1.37).
func newStorage(t *testing.T, authz authorizer.UnconditionalAuthorizer, clock clock.PassiveClock, usePKCS10 bool) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	resource := certificates.SchemeGroupVersion.WithResource("podcertificaterequests").GroupResource()
	var etcdStorage *storagebackend.ConfigForResource
	var server *etcd3testing.EtcdTestServer
	if usePKCS10 {
		etcdStorage, server = registrytest.NewEtcdStorageForResource(t, resource)
	} else {
		effectiveVersion := basecompatibility.NewEffectiveVersionFromString("1.39", "", "")
		effectiveVersion.SetEmulationVersion(version.MajorMinor(1, 36))
		effectiveVersion.SetMinCompatibilityVersion(version.MajorMinor(1, 36))
		etcdStorage, server = registrytest.NewEtcdStorageForResourceEffectiveVersion(t, resource, &certificates.PodCertificateRequest{}, effectiveVersion)
	}
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

// forEachSpecVariant runs fn once for a PCR with StubPKCS10Request (stored as
// v1) and once for a PCR with the deprecated PKIXPublicKey and
// ProofOfPossession fields (stored as v1beta1).
func forEachSpecVariant(t *testing.T, fn func(t *testing.T, usePKCS10 bool)) {
	for name, usePKCS10 := range map[string]bool{
		"pkcs10":                 true,
		"pkix-and-proof-v1beta1": false,
	} {
		t.Run(name, func(t *testing.T) {
			fn(t, usePKCS10)
		})
	}
}

func TestCreate(t *testing.T) {
	forEachSpecVariant(t, testCreate)
}

func testCreate(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	invalidPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	if usePKCS10 {
		invalidPCRwithPKCS10.Spec.StubPKCS10Request = []byte("invalid-csr-bytes")
	} else {
		_, _, otherPKIX, otherProof := mustMakeEd25519KeyAndProof(t, []byte("other-value"))
		invalidPCRwithPKCS10.Spec.PKIXPublicKey = otherPKIX      //nolint:staticcheck // Deprecated fields must be tested
		invalidPCRwithPKCS10.Spec.ProofOfPossession = otherProof //nolint:staticcheck // Deprecated fields must be tested
	}
	test.TestCreate(validPCRwithPKCS10, invalidPCRwithPKCS10)
}

func TestUpdate(t *testing.T) {
	forEachSpecVariant(t, testUpdate)
}

func testUpdate(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	// Define update functions
	validUpdate := func(object runtime.Object) runtime.Object {
		pcr := object.(*certificates.PodCertificateRequest)
		if pcr.ObjectMeta.Annotations == nil {
			pcr.ObjectMeta.Annotations = map[string]string{}
		}
		pcr.ObjectMeta.Annotations["k8s.io/cool-annotation"] = "value"
		return pcr
	}
	invalidUpdate := func(object runtime.Object) runtime.Object {
		pcr := object.(*certificates.PodCertificateRequest)
		pcr.Spec.SignerName = "test.k8s.io/new-signer"
		return pcr
	}

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestUpdate(validPCRwithPKCS10, validUpdate, invalidUpdate)
}

func TestUpdateStompsStatus(t *testing.T) {
	forEachSpecVariant(t, testUpdateStompsStatus)
}

func testUpdateStompsStatus(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	stompStatus := func(object runtime.Object) runtime.Object {
		pcr := object.(*certificates.PodCertificateRequest)
		// The strategy should stomp status.
		pcr.Status.NotAfter = new(metav1.NewTime(mustParseTime(t, "2025-01-01T00:00:00Z")))
		return pcr
	}

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestUpdate(validPCRwithPKCS10, stompStatus)
}

func TestUpdateStatus(t *testing.T) {
	forEachSpecVariant(t, testUpdateStatus)
}

func testUpdateStatus(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store, "status")
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	caCertDER, caPrivKey := mustMakeCA(t)
	podUID1 := types.UID("pod-uid-1")

	invalidUpdate := func(object runtime.Object) runtime.Object {
		pcr := object.(*certificates.PodCertificateRequest)
		pcr.Spec.SignerName = "test.k8s.io/new-signer"
		return pcr
	}

	validPCRwithPKCS10, pubKeyPKCS10 := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	pod1CertPKCS10 := mustSignCertForPublicKey(t, 24*time.Hour, pubKeyPKCS10, caCertDER, caPrivKey)
	test.TestUpdate(
		validPCRwithPKCS10,
		makeStatusUpdate(t, pod1CertPKCS10),
		invalidUpdate,
	)
}

func TestUpdateStatusStompsSpec(t *testing.T) {
	forEachSpecVariant(t, testUpdateStatusStompsSpec)
}

func testUpdateStatusStompsSpec(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store, "status")
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	stompSpec := func(object runtime.Object) runtime.Object {
		pcr := object.(*certificates.PodCertificateRequest)
		// The strategy should stomp spec.
		pcr.Spec.SignerName = "foo.com/bar"
		return pcr
	}

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestUpdate(validPCRwithPKCS10, stompSpec)
}

func TestUpdateStatusFailsWhenAuthorizerDenies(t *testing.T) {
	forEachSpecVariant(t, testUpdateStatusFailsWhenAuthorizerDenies)
}

func testUpdateStatusFailsWhenAuthorizerDenies(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionNoOpinion,
	}
	_, statusStorage, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store, "status")
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	caCertDER, caPrivKey := mustMakeCA(t)
	podUID1 := types.UID("pod-uid-1")

	noopUpdate := func(object runtime.Object) runtime.Object {
		return object
	}

	validPCRwithPKCS10, pubKeyPKCS10 := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	pod1CertPKCS10 := mustSignCertForPublicKey(t, 24*time.Hour, pubKeyPKCS10, caCertDER, caPrivKey)
	test.TestUpdate(
		validPCRwithPKCS10,
		noopUpdate,
		makeStatusUpdate(t, pod1CertPKCS10),
	)
}

func TestDelete(t *testing.T) {
	forEachSpecVariant(t, testDelete)
}

func testDelete(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestDelete(validPCRwithPKCS10)
}

func TestGet(t *testing.T) {
	forEachSpecVariant(t, testGet)
}

func testGet(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestGet(validPCRwithPKCS10)
}

func TestList(t *testing.T) {
	forEachSpecVariant(t, testList)
}

func testList(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestList(validPCRwithPKCS10)
}

func TestWatch(t *testing.T) {
	forEachSpecVariant(t, testWatch)
}

func testWatch(t *testing.T, usePKCS10 bool) {
	authz := &fakeAuthorizer{
		decision: authorizer.DecisionAllow,
	}
	storage, _, server := newStorage(t, authz, testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z")), usePKCS10)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test.SetUserInfo(&user.DefaultInfo{
		Name: "foo",
	})

	podUID1 := types.UID("pod-uid-1")

	validPCRwithPKCS10, _ := makeValidPCR(t, "foo2", test.TestNamespace(), podUID1, usePKCS10)
	test.TestWatch(
		validPCRwithPKCS10,
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
				"metadata.name":      "foo2",
				"spec.signerName":    "k8s.io/foo",
				"spec.nodeName":      "node-1",
			},
		},
		// not matching fields
		[]fields.Set{
			{
				"metadata.namespace": test.TestNamespace(),
				"metadata.name":      "foo2",
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

func mustMakePKCS10CSR(t *testing.T) (crypto.PublicKey, []byte) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating ed25519 key: %v", err)
	}
	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName: "foo",
		},
	}
	csrDER, err := x509.CreateCertificateRequest(rand.Reader, template, priv)
	if err != nil {
		t.Fatalf("Error while creating certificate request: %v", err)
	}
	return pub, csrDER
}

func makeValidPCR(t *testing.T, name string, namespace string, podUID types.UID, usePKCS10 bool) (*certificates.PodCertificateRequest, crypto.PublicKey) {
	pcr := &certificates.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: certificates.PodCertificateRequestSpec{
			SignerName:                "k8s.io/foo",
			PodName:                   "pod-1",
			PodUID:                    podUID,
			ServiceAccountName:        "sa-1",
			ServiceAccountUID:         types.UID("sa-uid-1"),
			NodeName:                  "node-1",
			NodeUID:                   types.UID("node-uid-1"),
			MaxExpirationSeconds:      new(int32(86400)),
			UnverifiedUserAnnotations: map[string]string{"test/foo": "bar"},
		},
	}
	var pubKey crypto.PublicKey
	if usePKCS10 {
		var validCSR []byte
		pubKey, validCSR = mustMakePKCS10CSR(t)
		pcr.Spec.StubPKCS10Request = validCSR
	} else {
		_, pubKeyPKIX, pubPKIX, sig := mustMakeEd25519KeyAndProof(t, []byte(podUID))
		pcr.Spec.PKIXPublicKey = pubPKIX //nolint:staticcheck // Deprecated fields must be tested
		pcr.Spec.ProofOfPossession = sig //nolint:staticcheck // Deprecated fields must be tested
		pubKey = pubKeyPKIX
	}
	return pcr, pubKey
}

func makeStatusUpdate(t *testing.T, certChain string) func(runtime.Object) runtime.Object {
	return func(object runtime.Object) runtime.Object {
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
			CertificateChain: certChain,
			NotBefore:        new(metav1.NewTime(mustParseTime(t, "1970-01-01T00:00:00Z"))),
			BeginRefreshAt:   new(metav1.NewTime(mustParseTime(t, "1970-01-01T12:00:00Z"))),
			NotAfter:         new(metav1.NewTime(mustParseTime(t, "1970-01-02T00:00:00Z"))),
		}
		return pcr
	}
}
