package podcertificate

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	mathrand "math/rand"
	"sync"
	"time"

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

type PodIdentity struct {
	Namespace          string
	PodName            string
	PodUID             string
	ServiceAccountName string
}

type Manager interface {
	GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, error)
	ForgetPodCertificateCredentialBundle(ctx context.Context, podUID, volumeName, path string)
}

type IssuingManager struct {
	kc         kubernetes.Interface
	nodeLister corelisters.NodeLister
	nodeName   types.NodeName

	clock clock.Clock

	lock        sync.Mutex
	credentials map[credentialKey]*credentialRecord
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, nodeLister corelisters.NodeLister, nodeName types.NodeName, clock clock.Clock) *IssuingManager {
	return &IssuingManager{
		kc:          kc,
		nodeLister:  nodeLister,
		nodeName:    nodeName,
		clock:       clock,
		credentials: map[credentialKey]*credentialRecord{},
	}
}

type credentialKey struct {
	podUID     string
	volumeName string
	path       string
}

type credentialRecord struct {
	credentialBundle []byte
	beginRefreshAt   time.Time
	expiresAt        time.Time
}

type credentialStatus int

const (
	credentialStatusAbsent credentialStatus = iota
	credentialStatusExpired
	credentialStatusStale
	credentialStatusFresh
)

func (m *IssuingManager) GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	record, status := m.checkStoredCredentials(pod.PodUID, volumeName, path)
	switch status {
	case credentialStatusFresh:
		// If we are fresh, return the cached content.
		return record.credentialBundle, nil
	case credentialStatusStale:
		// If we are stale, then run certificate issuance.  In case
		// of error, we still return the cached credential bundle.
		newRecord, err := m.makeCredentialBundle(ctx, signerName, pod, keyType)
		if err != nil {
			klog.ErrorS(err, "Failed to refresh workload certificate.  Will return existing stale certificate.", "namespace", pod.Namespace, "pod-name", pod.PodName, "pod-uid", pod.PodUID, "volume-name", volumeName, "path", path, "signer-name", signerName)
			return record.credentialBundle, nil
		}
		m.setStoredCredentials(pod.PodUID, volumeName, path, newRecord)
		return newRecord.credentialBundle, nil
	case credentialStatusExpired:
		// If we are expired, then run certificate issuance.  Errors are hard
		// --- it's better for the pod to go unhealthy than to continue running
		// with an expired certificate.
		newRecord, err := m.makeCredentialBundle(ctx, signerName, pod, keyType)
		if err != nil {
			klog.ErrorS(err, "Failed to refresh workload certificate.  Existing credentials are expired, so returning error.", "namespace", pod.Namespace, "pod-name", pod.PodName, "pod-uid", pod.PodUID, "volume-name", volumeName, "path", path, "signer-name", signerName)
			return nil, fmt.Errorf("while requesting new certificate (existing certificate is expired): %w", err)
		}
		m.setStoredCredentials(pod.PodUID, volumeName, path, newRecord)
		return newRecord.credentialBundle, nil
	case credentialStatusAbsent:
		// If we don't yet have any credentials, then run certificate issuance.
		// Errors are hard --- we want to block pod startup until it has a valid
		// certificate.
		newRecord, err := m.makeCredentialBundle(ctx, signerName, pod, keyType)
		if err != nil {
			klog.ErrorS(err, "Failed to refresh workload certificate.  There are no existing credentials, so returning error.", "namespace", pod.Namespace, "pod-name", pod.PodName, "pod-uid", pod.PodUID, "volume-name", volumeName, "path", path, "signer-name", signerName)
			return nil, fmt.Errorf("while requesting initial certificate: %w", err)
		}
		m.setStoredCredentials(pod.PodUID, volumeName, path, newRecord)
		return newRecord.credentialBundle, nil
	default:
		panic(fmt.Sprintf("Unknown credentialStatus enum value %d", status))
	}
}

func (m *IssuingManager) checkStoredCredentials(podUID, volumeName, path string) (*credentialRecord, credentialStatus) {
	m.lock.Lock()
	defer m.lock.Unlock()

	record, ok := m.credentials[credentialKey{
		podUID:     podUID,
		volumeName: volumeName,
		path:       path,
	}]
	if !ok {
		return nil, credentialStatusAbsent
	}

	now := m.clock.Now()
	if now.After(record.expiresAt) {
		return record, credentialStatusExpired
	}
	if now.After(record.beginRefreshAt) {
		return record, credentialStatusStale
	}
	return record, credentialStatusFresh
}

func (m *IssuingManager) setStoredCredentials(podUID, volumeName, path string, record *credentialRecord) {
	m.lock.Lock()
	defer m.lock.Unlock()

	key := credentialKey{
		podUID:     podUID,
		volumeName: volumeName,
		path:       path,
	}
	m.credentials[key] = record
}

// makeCredentialBundle is the entry point for really making a new credential bundle.
func (m *IssuingManager) makeCredentialBundle(ctx context.Context, signerName string, pod PodIdentity, keyType string) (*credentialRecord, error) {
	node, err := m.nodeLister.Get(string(m.nodeName))
	if err != nil {
		return nil, fmt.Errorf("while getting node object from local cache: %w", err)
	}

	serviceAccount, err := m.kc.CoreV1().ServiceAccounts(pod.Namespace).Get(ctx, pod.ServiceAccountName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("while fetching service account: %w", err)
	}

	privateKey, publicKey, proof, err := m.generateKeyAndProof(keyType, []byte(pod.PodUID))
	if err != nil {
		return nil, fmt.Errorf("while generating keypair: %w", err)
	}

	pkixPublicKey, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("while marshaling public key: %w", err)
	}

	keyPEM, err := pemEncodeKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("while PEM-encoding private key: %w", err)
	}

	req := &certificatesv1alpha1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    pod.Namespace,
			GenerateName: "req-" + pod.PodName,
		},
		Spec: certificatesv1alpha1.PodCertificateRequestSpec{
			SignerName:         signerName,
			PodName:            pod.PodName,
			PodUID:             types.UID(pod.PodUID),
			ServiceAccountName: pod.ServiceAccountName,
			ServiceAccountUID:  serviceAccount.ObjectMeta.UID,
			NodeName:           m.nodeName,
			NodeUID:            node.ObjectMeta.UID,
			PKIXPublicKey:      pkixPublicKey,
			ProofOfPossession:  proof,
		},
	}

	req, err = m.kc.CertificatesV1alpha1().PodCertificateRequests(pod.Namespace).Create(ctx, req, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("while creating PodCertificateRequest: %w", err)
	}

	req, err = WaitForPodCertificateRequest(ctx, m.kc, req.ObjectMeta.Namespace, req.ObjectMeta.Name, req.ObjectMeta.UID)
	if err != nil {
		return nil, fmt.Errorf("while waiting for PodCertificateRequest to be issued: %w", err)
	}

	// TODO(ahmedtd): Do we need any of these lifetime checks?  If we have them,
	// they should probably be enforced by kube-apiserver.
	leafCert, err := x509.ParseCertificate(req.Status.CertificateChain[0])
	if err != nil {
		return nil, fmt.Errorf("while parsing leaf certificate: %w", err)
	}
	if leafCert.NotBefore.After(m.clock.Now()) {
		return nil, fmt.Errorf("signer issued certificate that is not yet valid: %w", err)
	}
	if leafCert.NotAfter.Before(m.clock.Now()) {
		return nil, fmt.Errorf("signer issued certificate that has already expired: %w", err)
	}
	lifetime := leafCert.NotAfter.Sub(leafCert.NotBefore)
	if lifetime < 1*time.Hour {
		return nil, fmt.Errorf("signer issued certificate with a lifetime shorter than one hour")
	}

	// Assemble credential bundle from key and cert chain.
	credentialBundle := bytes.Buffer{}
	credentialBundle.Write(keyPEM)
	for _, cert := range req.Status.CertificateChain {
		credentialBundle.Write(pem.EncodeToMemory(&pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert,
		}))
	}

	// Always jitter the refresh time by up to 5 minutes to prevent volumes from
	// synchronizing their refresh times.
	beginRefreshAt := req.Status.BeginRefreshAt.Time.Add(-time.Duration(mathrand.Intn(5*60*1000)) * time.Millisecond)

	record := &credentialRecord{
		credentialBundle: credentialBundle.Bytes(),
		expiresAt:        leafCert.NotAfter,
		beginRefreshAt:   beginRefreshAt,
	}

	return record, nil
}

func (m *IssuingManager) generateKeyAndProof(keyType string, toBeSigned []byte) (privKey, pubKey any, sig []byte, err error) {
	hasher := crypto.SHA256.New()
	hasher.Write(toBeSigned)
	hash := hasher.Sum(nil)

	switch keyType {
	case "RSA2048":
		key, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 2048 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hash)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "RSA3072":
		key, err := rsa.GenerateKey(rand.Reader, 3072)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 3072 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hash)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "RSA4096":
		key, err := rsa.GenerateKey(rand.Reader, 4096)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 4096 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hash)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "", "ECDSAP256":
		key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating ECDSA P256 key: %w", err)
		}
		sig, err := ecdsa.SignASN1(rand.Reader, key, hash)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "ECDSAP384":
		key, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating ECDSA P384 key: %w", err)
		}
		sig, err := ecdsa.SignASN1(rand.Reader, key, hash)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	default:
		return nil, nil, nil, fmt.Errorf("unknown key type %q", keyType)
	}
}

// TODO(ahmedtd): Switch from one watch per req to using an informer on all reqs for node.
func WaitForPodCertificateRequest(ctx context.Context, client kubernetes.Interface, namespace, reqName string, reqUID types.UID) (*certificatesv1alpha1.PodCertificateRequest, error) {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", reqName).String()

	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return client.CertificatesV1alpha1().PodCertificateRequests(namespace).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return client.CertificatesV1alpha1().PodCertificateRequests(namespace).Watch(ctx, options)
		},
	}

	var issuedReq *certificatesv1alpha1.PodCertificateRequest
	_, err := watchtools.UntilWithSync(
		ctx,
		lw,
		&certificatesv1alpha1.PodCertificateRequest{},
		nil,
		func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified, watch.Added:
			case watch.Deleted:
				return false, fmt.Errorf("PodCertificateRequest %s/%s was deleted", namespace, reqName)
			default:
				return false, nil
			}

			req := event.Object.(*certificatesv1alpha1.PodCertificateRequest)

			if req.UID != reqUID {
				return false, fmt.Errorf("PodCertificateRequest %s/%s changed UIDs", namespace, reqName)
			}

			// Check for denied or failed.
			for _, c := range req.Status.Conditions {
				if c.Type == certificatesv1alpha1.PodCertificateRequestDenied {
					return false, fmt.Errorf("PodCertificateRequest %s/%s was denied", namespace, reqName)
				}
				if c.Type == certificatesv1alpha1.PodCertificateRequestFailed {
					return false, fmt.Errorf("PodCertificateRequest %s/%s was failed", namespace, reqName)
				}
			}

			// If the request hasn't had a certificate issued yet, keep watching.
			if len(req.Status.CertificateChain) == 0 {
				return false, nil
			}

			// The request was had a certificate issued.
			issuedReq = req
			return true, nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("while waiting for PodCertificateRequest: %w", err)
	}

	return issuedReq, nil
}

func pemEncodeKey(key any) ([]byte, error) {
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return nil, fmt.Errorf("while marshaling key to PKCS#8: %w", err)
	}

	return pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: keyDER,
	}), nil
}

func (m *IssuingManager) ForgetPodCertificateCredentialBundle(ctx context.Context, podUID, volumeName, path string) {
	m.lock.Lock()
	defer m.lock.Unlock()

	delete(m.credentials, credentialKey{
		podUID:     podUID,
		volumeName: volumeName,
		path:       path,
	})
}

type NoOpManager struct{}

func (m *NoOpManager) GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (m *NoOpManager) ForgetPodCertificateCredentialBundle(ctx context.Context, podUID, volumeName, path string) {
}
