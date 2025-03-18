package podcertificate

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"sync"
	"time"

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1alpha1 "k8s.io/client-go/informers/certificates/v1alpha1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// TODO(KEP-4317): Tests for IssuingManager

type PodIdentity struct {
	Namespace          string
	PodName            string
	PodUID             types.UID
	ServiceAccountName string
}

// PodManager is a local wrapper interface for pod.Manager.
type PodManager interface {
	// GetPods returns the regular pods bound to the kubelet and their spec.
	GetPods() []*corev1.Pod
}

type Manager interface {
	TrackPod(ctx context.Context, pod *corev1.Pod) error
	ForgetPod(ctx context.Context, pod *corev1.Pod)

	GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, []byte, error)
}

type IssuingManager struct {
	kc kubernetes.Interface

	podManager PodManager

	pcrInformer cache.SharedIndexInformer
	pcrLister   certlistersv1alpha1.PodCertificateRequestLister

	nodeLister corelisters.NodeLister
	nodeName   types.NodeName

	clock clock.WithTicker

	lock               sync.Mutex
	validCredentials   map[credentialKey]*validCredentialRecord
	invalidCredentials map[credentialKey]*invalidCredentialRecord
	// pendingCredentials maps namespace/name keys of PodCertificateRequests to
	// private key and volume information.
	pendingCredentials map[string]*pendingCredentialRecord
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, podManager PodManager, pcrInformer certinformersv1alpha1.PodCertificateRequestInformer, nodeLister corelisters.NodeLister, nodeName types.NodeName, clock clock.WithTicker) *IssuingManager {
	m := &IssuingManager{
		kc:                 kc,
		podManager:         podManager,
		pcrInformer:        pcrInformer.Informer(),
		pcrLister:          pcrInformer.Lister(),
		nodeLister:         nodeLister,
		nodeName:           nodeName,
		clock:              clock,
		validCredentials:   map[credentialKey]*validCredentialRecord{},
		invalidCredentials: map[credentialKey]*invalidCredentialRecord{},
		pendingCredentials: map[string]*pendingCredentialRecord{},
	}

	// Add informer functions.  We don't follow the typical controller pattern
	// where we use a workqueue, because we don't have to do any RPCs or state
	// mutations that can fail.  We just acquire a lock and do some map updates.
	m.pcrInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		// Nothing to do on add.
		UpdateFunc: m.handlePCRUpdate,
		DeleteFunc: m.handlePCRDelete,
	})

	return m
}

type credentialKey struct {
	podUID     types.UID
	volumeName string
	path       string
}

type validCredentialRecord struct {
	key            []byte
	certificates   []byte
	beginRefreshAt time.Time
	expiresAt      time.Time
}

type invalidCredentialRecord struct {
	condition string
	reason    string
	message   string
}

type pendingCredentialRecord struct {
	privateKey []byte
	podUID     types.UID
	volumeName string
	path       string
}

type credentialStatus int

const (
	credentialStatusAbsent credentialStatus = iota
	credentialStatusExpired
	credentialStatusStale
	credentialStatusFresh
)

func (m *IssuingManager) Run(ctx context.Context) {
	if !cache.WaitForCacheSync(ctx.Done(), m.pcrInformer.HasSynced) {
		return
	}
	go wait.JitterUntilWithContext(ctx, m.refreshCredentials, 1*time.Minute, 1.0, false)
	<-ctx.Done()
}

func (m *IssuingManager) refreshCredentials(ctx context.Context) {
	allPods := m.podManager.GetPods()
	for _, pod := range allPods {
		if err := m.refreshOnePod(ctx, pod); err != nil {
			klog.ErrorS(err, "Error while refreshing pod", "namespace", pod.ObjectMeta.Namespace, "name", pod.ObjectMeta.Name)
		}
	}
}

func (m *IssuingManager) refreshOnePod(ctx context.Context, pod *corev1.Pod) error {
	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for _, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			uniquePath := source.PodCertificate.CredentialBundlePath
			if uniquePath == "" {
				uniquePath = source.PodCertificate.KeyPath
			}
			if uniquePath == "" {
				uniquePath = source.PodCertificate.CertificateChainPath
			}

			stale := false
			func() {
				m.lock.Lock()
				defer m.lock.Unlock()
				credRecord, ok := m.validCredentials[credentialKey{pod.ObjectMeta.UID, v.Name, uniquePath}]
				if !ok {
					// Either we just created the PCR for this pod/volume/path
					// combo, and it hasn't been issued yet, or the initial
					// issuance was denied or failed.  Either way, no need to
					// refresh.
					return
				}
				if m.clock.Now().After(credRecord.beginRefreshAt) {
					stale = true
				}
			}()
			if !stale {
				continue
			}

			serviceAccount, err := m.kc.CoreV1().ServiceAccounts(pod.ObjectMeta.Namespace).Get(ctx, pod.Spec.ServiceAccountName, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("while fetching service account: %w", err)
			}

			node, err := m.nodeLister.Get(string(m.nodeName))
			if err != nil {
				return fmt.Errorf("while getting node object from local cache: %w", err)
			}

			err = m.createPCR(
				ctx,
				pod.ObjectMeta.Namespace,
				pod.ObjectMeta.Name, pod.ObjectMeta.UID,
				pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
				m.nodeName, node.ObjectMeta.UID,
				v.Name, uniquePath,
				source.PodCertificate.SignerName, source.PodCertificate.KeyType,
			)
			if err != nil {
				return fmt.Errorf("while creating pcr: %w", err)
			}
		}
	}

	return nil
}

// TrackPod visits all PodCertificate projected volume sources within the pod,
// creating an initial PodCertificateRequest for each one.
func (m *IssuingManager) TrackPod(ctx context.Context, pod *corev1.Pod) error {
	klog.InfoS("XXXX TrackPod entry", "pod", pod.ObjectMeta.Namespace+"/"+pod.ObjectMeta.Name)
	serviceAccount, err := m.kc.CoreV1().ServiceAccounts(pod.ObjectMeta.Namespace).Get(ctx, pod.Spec.ServiceAccountName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("while fetching service account: %w", err)
	}

	node, err := m.nodeLister.Get(string(m.nodeName))
	if err != nil {
		return fmt.Errorf("while getting node object from local cache: %w", err)
	}

	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for _, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			uniquePath := source.PodCertificate.CredentialBundlePath
			if uniquePath == "" {
				uniquePath = source.PodCertificate.KeyPath
			}

			err := m.createPCR(
				ctx,
				pod.ObjectMeta.Namespace,
				pod.ObjectMeta.Name, pod.ObjectMeta.UID,
				pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
				m.nodeName, node.ObjectMeta.UID,
				v.Name, uniquePath,
				source.PodCertificate.SignerName, source.PodCertificate.KeyType,
			)
			if err != nil {
				return fmt.Errorf("while creating pcr: %w", err)
			}
		}
	}

	klog.InfoS("XXXX TrackPod successful completion", "pod", pod.ObjectMeta.Namespace+"/"+pod.ObjectMeta.Name)
	return nil
}

// ForgetPod removes all live credential records associated with pod.
func (m *IssuingManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
	klog.InfoS("XXXX ForgetPod entry", "pod", pod.ObjectMeta.Namespace+"/"+pod.ObjectMeta.Name)
	m.lock.Lock()
	defer m.lock.Unlock()

	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for _, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			uniquePath := source.PodCertificate.CredentialBundlePath
			if uniquePath == "" {
				uniquePath = source.PodCertificate.KeyPath
			}

			key := credentialKey{
				podUID:     pod.ObjectMeta.UID,
				volumeName: v.Name,
				path:       uniquePath,
			}
			delete(m.validCredentials, key)
		}
	}

	klog.InfoS("XXXX ForgetPod successful completion", "pod", pod.ObjectMeta.Namespace+"/"+pod.ObjectMeta.Name)
}

// On update, check if the PodCertificateRequest has moved to a terminal state.
// If so, convert its pending credential record to a live credential record.
func (m *IssuingManager) handlePCRUpdate(old, new any) {
	key, err := cache.MetaNamespaceKeyFunc(new)
	if err != nil {
		return
	}
	newPCR := new.(*certificatesv1alpha1.PodCertificateRequest)

	klog.InfoS("Handling PodCertificateRequest update", "pcr", key)

	state := ""
	reason := ""
	message := ""
	for _, cond := range newPCR.Status.Conditions {
		// PCRs can have only one condition
		state = cond.Type
		reason = cond.Reason
		message = cond.Message
	}
	if state == "" {
		// PCR has not moved to a terminal state.  Do nothing.
		return
	}

	klog.InfoS("Checked state of PodCertificateRequest", "state", state)

	m.lock.Lock()
	defer m.lock.Unlock()

	// Clear the pending credential record and make a live credential
	// record.
	pendingRecord := m.pendingCredentials[key]
	delete(m.pendingCredentials, key)

	if pendingRecord == nil {
		// If we don't have a pending record corresponding to this PCR, there
		// are two possible causes:
		//
		// 1) Some admin manually created a PCR that targets a pod on our node.
		//
		// 2) Kubelet created this PCR, but then restarted.
		//
		// In both cases, just do nothing.
		klog.InfoS("PodCertificateRequest update was for stray", "pcr", key)
		return
	}

	finalKey := credentialKey{
		podUID:     newPCR.Spec.PodUID,
		volumeName: pendingRecord.volumeName,
		path:       pendingRecord.path,
	}
	if state == certificatesv1alpha1.PodCertificateRequestConditionTypeIssued {
		klog.InfoS("Recording valid credential entry", "poduid", newPCR.Spec.PodUID, "volumename", pendingRecord.volumeName, "path", pendingRecord.path)
		m.validCredentials[finalKey] = &validCredentialRecord{
			key:            pendingRecord.privateKey,
			certificates:   []byte(newPCR.Status.CertificateChain),
			beginRefreshAt: newPCR.Status.BeginRefreshAt.Time,
			expiresAt:      newPCR.Status.NotAfter.Time,
		}
	} else {
		klog.InfoS("Recording invalid credential entry", "poduid", newPCR.Spec.PodUID, "volumename", pendingRecord.volumeName, "path", pendingRecord.path)
		m.invalidCredentials[finalKey] = &invalidCredentialRecord{
			condition: state,
			reason:    reason,
			message:   message,
		}
	}
}

// On delete, clean up the pending credential record corresponding
// to this PodCertificateRequest, if any exists.
func (m *IssuingManager) handlePCRDelete(old any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(old)
	if err != nil {
		return
	}
	m.lock.Lock()
	defer m.lock.Unlock()
	delete(m.pendingCredentials, key)
}

// createPCR creates a PodCertificateRequest and records it in the
// pendingCredentials map.  The PodCertificateRequest informer loop will then
// wait for it to be issued, denied, or failed, and update liveCredentials.
func (m *IssuingManager) createPCR(
	ctx context.Context,
	namespace, podName string, podUID types.UID, serviceAccountName string, serviceAccountUID types.UID, nodeName types.NodeName, nodeUID types.UID, volumeName, path, signerName, keyType string) error {

	privateKey, publicKey, proof, err := m.generateKeyAndProof(keyType, []byte(podUID))
	if err != nil {
		return fmt.Errorf("while generating keypair: %w", err)
	}

	pkixPublicKey, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return fmt.Errorf("while marshaling public key: %w", err)
	}

	keyPEM, err := pemEncodeKey(privateKey)
	if err != nil {
		return fmt.Errorf("while PEM-encoding private key: %w", err)
	}

	req := &certificatesv1alpha1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    namespace,
			GenerateName: "req-" + podName,
		},
		Spec: certificatesv1alpha1.PodCertificateRequestSpec{
			SignerName:         signerName,
			PodName:            podName,
			PodUID:             podUID,
			ServiceAccountName: serviceAccountName,
			ServiceAccountUID:  serviceAccountUID,
			NodeName:           nodeName,
			NodeUID:            nodeUID,
			PKIXPublicKey:      pkixPublicKey,
			ProofOfPossession:  proof,
		},
	}

	req, err = m.kc.CertificatesV1alpha1().PodCertificateRequests(namespace).Create(ctx, req, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("while creating on API: %w", err)
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	pendingKey := req.ObjectMeta.Namespace + "/" + req.ObjectMeta.Name
	m.pendingCredentials[pendingKey] = &pendingCredentialRecord{
		privateKey: keyPEM,
		podUID:     podUID,
		volumeName: volumeName,
		path:       path,
	}

	return nil
}

func (m *IssuingManager) GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, []byte, error) {
	m.lock.Lock()
	defer m.lock.Unlock()

	key := credentialKey{
		podUID:     pod.PodUID,
		volumeName: volumeName,
		path:       path,
	}

	record := m.validCredentials[key]
	if record != nil {
		return record.key, record.certificates, nil
	}

	invalidRecord := m.invalidCredentials[key]
	if invalidRecord != nil {
		return nil, nil, fmt.Errorf("PodCertificateRequest was rejected with condition=%s reason=%s message=%q", invalidRecord.condition, invalidRecord.reason, invalidRecord.message)
	}

	return nil, nil, fmt.Errorf("no credentials yet")
}

func hashBytes(in []byte) []byte {
	out := sha256.Sum256(in)
	return out[:]
}

func (m *IssuingManager) generateKeyAndProof(keyType string, toBeSigned []byte) (privKey crypto.PrivateKey, pubKey crypto.PublicKey, sig []byte, err error) {
	switch keyType {
	case "RSA2048":
		key, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 2048 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hashBytes(toBeSigned))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "RSA3072":
		key, err := rsa.GenerateKey(rand.Reader, 3072)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 3072 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hashBytes(toBeSigned))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "RSA4096":
		key, err := rsa.GenerateKey(rand.Reader, 4096)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 4096 key: %w", err)
		}
		sig, err := rsa.SignPKCS1v15(nil, key, crypto.SHA256, hashBytes(toBeSigned))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "ECDSAP256":
		key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating ECDSA P256 key: %w", err)
		}
		sig, err := ecdsa.SignASN1(rand.Reader, key, hashBytes(toBeSigned))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "ECDSAP384":
		key, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating ECDSA P384 key: %w", err)
		}
		sig, err := ecdsa.SignASN1(rand.Reader, key, hashBytes(toBeSigned))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "ED25519":
		pub, priv, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating Ed25519 key: %w", err)
		}
		sig := ed25519.Sign(priv, toBeSigned)
		return priv, pub, sig, nil
	default:
		return nil, nil, nil, fmt.Errorf("unknown key type %q", keyType)
	}
}

func pemEncodeKey(key crypto.PrivateKey) ([]byte, error) {
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return nil, fmt.Errorf("while marshaling key to PKCS#8: %w", err)
	}

	return pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: keyDER,
	}), nil
}

type NoOpManager struct{}

var _ Manager = (*NoOpManager)(nil)

func (m *NoOpManager) TrackPod(ctx context.Context, pod *corev1.Pod) error {
	return nil
}

func (m *NoOpManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) GetPodCertificateCredentialBundle(ctx context.Context, pod PodIdentity, volumeName, path, signerName, keyType string) ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("unimplemented")
}
