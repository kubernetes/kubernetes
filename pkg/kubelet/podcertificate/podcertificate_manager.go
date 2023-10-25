package podcertificate

import (
	"context"
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

	certificatesv1 "k8s.io/api/certificates/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	utilcert "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/component-helpers/kubernetesx509"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

type PodIdentity struct {
	Namespace          string
	PodName            string
	PodUID             string
	ServiceAccountName string
	NodeName           string
}

type Manager interface {
	GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, error)
	ForgetPodCertificateCredentialBundle(ctx context.Context, podUID, volumeName, path string)
}

type IssuingManager struct {
	kc    kubernetes.Interface
	clock clock.Clock

	lock        sync.Mutex
	credentials map[credentialKey]*credentialRecord
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, clock clock.Clock) *IssuingManager {
	return &IssuingManager{
		kc:          kc,
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
	privateKey, err := m.generateKey(keyType)
	if err != nil {
		return nil, err
	}

	keyPEM, err := pemEncodeKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("while PEM-encoding private key: %w", err)
	}

	crTemplate := &x509.CertificateRequest{}

	podIdentity := &kubernetesx509.PodIdentity{
		Namespace:          pod.Namespace,
		PodName:            pod.PodName,
		PodUID:             pod.PodUID,
		ServiceAccountName: pod.ServiceAccountName,
		NodeName:           pod.NodeName,
	}
	kubernetesx509.AddPodIdentityToCertificateRequest(podIdentity, crTemplate)

	csrPEM, err := utilcert.MakeCSRFromTemplate(privateKey, crTemplate)
	if err != nil {
		return nil, fmt.Errorf("while making serialized X.509 CSR: %w", err)
	}

	csrObj := &certificatesv1.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "workload-certificates-",
			Labels: map[string]string{
				"debug.workloadcertificates.kubelet.kubernetes.io/pod-namespace":        pod.Namespace,
				"debug.workloadcertificates.kubelet.kubernetes.io/pod-name":             pod.PodName,
				"debug.workloadcertificates.kubelet.kubernetes.io/pod-uid":              pod.PodUID,
				"debug.workloadcertificates.kubelet.kubernetes.io/service-account-name": pod.ServiceAccountName,
			},
		},
		Spec: certificatesv1.CertificateSigningRequestSpec{
			SignerName: signerName,
			Request:    csrPEM,
			Usages: []certificatesv1.KeyUsage{
				certificatesv1.UsageDigitalSignature,
				certificatesv1.UsageServerAuth,
				certificatesv1.UsageClientAuth,
				certificatesv1.UsageKeyAgreement,
				certificatesv1.UsageKeyEncipherment,
			},
		},
	}

	csrObj, err = m.kc.CertificatesV1().CertificateSigningRequests().Create(ctx, csrObj, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("while creating CertificateSigningRequest on API server: %w", err)
	}

	certChainPEM, err := csr.WaitForCertificate(ctx, m.kc, csrObj.ObjectMeta.Name, csrObj.ObjectMeta.UID)
	if err != nil {
		return nil, fmt.Errorf("while waiting for CertificateSigningRequest to be approved and issued: %w", err)
	}

	// Assume that the first certificate in the returned chain is the leaf
	// certificate.  Parse it to extract the expiry time.
	leafBlock, _ := pem.Decode(certChainPEM)
	if leafBlock == nil {
		return nil, fmt.Errorf("no block found when parsing issued certificate chain")
	}
	leafCert, err := x509.ParseCertificate(leafBlock.Bytes)
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

	// Refresh at the lesser of 24 hours or half of the certificate lifetime.
	// Always jitter by up to 5 minutes to prevent volumes from synchronizing
	// their refresh times.
	beginRefreshAt := leafCert.NotBefore.Add(24*time.Hour - time.Duration(mathrand.Intn(5*60*1000))*time.Millisecond)
	if lifetime/2 < 24*time.Hour {
		beginRefreshAt = leafCert.NotBefore.Add(lifetime/2 - time.Duration(mathrand.Intn(5*60*1000))*time.Millisecond)
	}

	record := &credentialRecord{
		credentialBundle: append(keyPEM, certChainPEM...),
		expiresAt:        leafCert.NotAfter,
		beginRefreshAt:   beginRefreshAt,
	}

	return record, nil
}

func (m *IssuingManager) generateKey(keyType string) (key any, err error) {
	switch keyType {
	case "RSA2048":
		key, err = rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, fmt.Errorf("while generating RSA 2048 key: %w", err)
		}
		return key, nil
	case "RSA3072":
		key, err = rsa.GenerateKey(rand.Reader, 3072)
		if err != nil {
			return nil, fmt.Errorf("while generating RSA 3072 key: %w", err)
		}
		return key, nil
	case "RSA4096":
		key, err = rsa.GenerateKey(rand.Reader, 4096)
		if err != nil {
			return nil, fmt.Errorf("while generating RSA 4096 key: %w", err)
		}
		return key, nil
	case "", "ECDSAP256":
		key, err = ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			return nil, fmt.Errorf("while generating ECDSA P256 key: %w", err)
		}
		return key, nil
	case "ECDSAP384":
		key, err = ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		if err != nil {
			return nil, fmt.Errorf("while generating ECDSA P384 key: %w", err)
		}
		return key, nil
	default:
		return nil, fmt.Errorf("unknown key type %q", keyType)
	}
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
