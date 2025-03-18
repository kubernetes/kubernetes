package storage

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
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
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	testclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, certificates.SchemeGroupVersion.WithResource("podcertificaterequests").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "podcertificaterequests",
	}
	storage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, statusStorage, server
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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
		// Invalid PCR -- proof-of-possession signed wrong value
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
				PKIXPublicKey:        ed25519PubPKIX2,
				ProofOfPossession:    ed25519Proof2,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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

func TestUpdateStatus(t *testing.T) {
	// It's pretty gross that the strategy is a global object.
	podcertificaterequest.StatusStrategy.Clock = testclock.NewFakePassiveClock(mustParseTime(t, "1970-01-01T00:00:00Z"))

	_, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer statusStorage.store.DestroyFunc()

	test := genericregistrytest.New(t, statusStorage.store)

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
						Type:    certificates.PodCertificateRequestConditionTypeIssued,
						Status:  metav1.ConditionTrue,
						Reason:  "Whatever",
						Message: "Foo message",
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

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)

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

func mustMakeIntermediateCA(t *testing.T, rootDER []byte, rootPrivateKey crypto.PrivateKey) ([]byte, ed25519.PrivateKey) {
	intermediatePub, intermediatePriv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating intermediate signing key: %v", err)
	}

	intermediateCertTemplate := &x509.Certificate{
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		NotBefore:             mustParseTime(t, "1970-01-01T00:00:00Z"),
		NotAfter:              mustParseTime(t, "1971-01-01T00:00:00Z"),
	}

	rootCert, err := x509.ParseCertificate(rootDER)
	if err != nil {
		t.Fatalf("Error while parsing root certificate: %v", err)
	}

	intermediateCertDER, err := x509.CreateCertificate(rand.Reader, intermediateCertTemplate, rootCert, intermediatePub, rootPrivateKey)
	if err != nil {
		t.Fatalf("Error while creating intermediate certificate: %v", err)
	}

	return intermediateCertDER, intermediatePriv
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

func mustMakeECDSAKeyAndProof(t *testing.T, curve elliptic.Curve, toBeSigned []byte) (*ecdsa.PrivateKey, *ecdsa.PublicKey, []byte, []byte) {
	priv, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatalf("Error while generating ECDSA key: %v", err)
	}
	pubPKIX, err := x509.MarshalPKIXPublicKey(priv.Public())
	if err != nil {
		t.Fatalf("Error while marshaling PKIX public key: %v", err)
	}
	sig, err := ecdsa.SignASN1(rand.Reader, priv, hashBytes(toBeSigned))
	if err != nil {
		t.Fatalf("Error while making proof of possession: %v", err)
	}
	return priv, &priv.PublicKey, pubPKIX, sig
}

func mustMakeRSAKeyAndProof(t *testing.T, modulusSize int, toBeSigned []byte) (*rsa.PrivateKey, *rsa.PublicKey, []byte, []byte) {
	priv, err := rsa.GenerateKey(rand.Reader, modulusSize)
	if err != nil {
		t.Fatalf("Error while generating RSA key: %v", err)
	}
	pubPKIX, err := x509.MarshalPKIXPublicKey(&priv.PublicKey)
	if err != nil {
		t.Fatalf("Error while marshaling public key: %v", err)
	}
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, hashBytes(toBeSigned))
	if err != nil {
		t.Fatalf("Error while making proof of possession: %v", err)
	}
	return priv, &priv.PublicKey, pubPKIX, sig
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

func pemEncode(blockType string, data []byte) string {
	return string(pem.EncodeToMemory(&pem.Block{
		Type:  blockType,
		Bytes: data,
	}))
}

func hashBytes(in []byte) []byte {
	out := sha256.Sum256(in)
	return out[:]
}
