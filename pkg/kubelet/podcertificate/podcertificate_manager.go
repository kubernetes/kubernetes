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
	coreinformersv1 "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// PodManager is a local wrapper interface for pod.Manager.
type PodManager interface {
	// GetPods returns the regular pods bound to the kubelet and their spec.
	GetPods() []*corev1.Pod
}

// Manager abstracts the functionality needed by Kubelet and the volume host in
// order to provide pod certificate functionality.
type Manager interface {
	// TrackPod is called by Kubelet every time a new pod is assigned to the node.
	TrackPod(ctx context.Context, pod *corev1.Pod) error
	// ForgetPod is called by Kubelet every time a pod is dropped from the node.
	ForgetPod(ctx context.Context, pod *corev1.Pod)

	// GetPodCertificateCredentialBundle is called by the volume host to
	// retrieve the creential bundle for a given pod certificate volume.
	GetPodCertificateCredentialBundle(ctx context.Context, podUID types.UID, volumeName string, sourceIndex int) ([]byte, []byte, error)
}

// IssuingManager is the main implementation of Manager.
type IssuingManager struct {
	kc kubernetes.Interface

	podManager PodManager

	pcrInformer cache.SharedIndexInformer
	pcrLister   certlistersv1alpha1.PodCertificateRequestLister

	nodeInformer cache.SharedIndexInformer
	nodeLister   corelistersv1.NodeLister
	nodeName     types.NodeName

	clock clock.WithTicker

	// lock covers liveCredentials and prendingCredentials
	lock            sync.Mutex
	liveCredentials map[credentialKey]*liveCredentialRecord
	// pendingCredentials holds onto the private keys associated with in-flight
	// PodCertificateRequests.
	pendingCredentials map[types.UID]*pendingCredentialRecord
}

type pendingCredentialRecord struct {
	privateKey                 []byte
	projectedVolumeName        string
	projectedVolumeSourceIndex int
}

type credentialKey struct {
	podUID                     types.UID
	projectedVolumeName        string
	projectedVolumeSourceIndex int
}

type liveCredentialRecord struct {
	privateKey []byte
	pcr        *certificatesv1alpha1.PodCertificateRequest
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, podManager PodManager, pcrInformer certinformersv1alpha1.PodCertificateRequestInformer, nodeInformer coreinformersv1.NodeInformer, nodeName types.NodeName, clock clock.WithTicker) *IssuingManager {
	m := &IssuingManager{
		kc:           kc,
		podManager:   podManager,
		pcrInformer:  pcrInformer.Informer(),
		pcrLister:    pcrInformer.Lister(),
		nodeInformer: nodeInformer.Informer(),
		nodeLister:   nodeInformer.Lister(),
		nodeName:     nodeName,
		clock:        clock,

		liveCredentials:    map[credentialKey]*liveCredentialRecord{},
		pendingCredentials: map[types.UID]*pendingCredentialRecord{},
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

func (m *IssuingManager) Run(ctx context.Context) {
	klog.InfoS("podcertificate.IssuingManager starting up")
	if !cache.WaitForCacheSync(ctx.Done(), m.pcrInformer.HasSynced, m.nodeInformer.HasSynced) {
		return
	}
	go wait.JitterUntilWithContext(ctx, m.runRefreshPass, 1*time.Minute, 1.0, false)
	<-ctx.Done()
	klog.InfoS("podcertificate.IssuingManager shut down")
}

func (m *IssuingManager) runRefreshPass(ctx context.Context) {
	allPods := m.podManager.GetPods()
	for _, pod := range allPods {
		if err := m.refreshOnePod(ctx, pod); err != nil {
			klog.ErrorS(err, "Error while refreshing pod", "namespace", pod.ObjectMeta.Namespace, "name", pod.ObjectMeta.Name)
		}
	}
}

func (m *IssuingManager) refreshOnePod(ctx context.Context, pod *corev1.Pod) error {
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

		for sourceIndex, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			stale := false
			func() {
				m.lock.Lock()
				defer m.lock.Unlock()
				credRecord, ok := m.liveCredentials[credentialKey{pod.ObjectMeta.UID, v.Name, sourceIndex}]
				if !ok {
					// Either we just created the PCR for this pod/volume/path
					// combo, and it hasn't been issued yet, or the initial
					// issuance was denied or failed.  Either way, no need to
					// refresh.
					return
				}
				if m.clock.Now().After(credRecord.pcr.Status.BeginRefreshAt.Time) {
					stale = true
				}
			}()
			if !stale {
				continue
			}

			err = m.createPCR(
				ctx,
				pod.ObjectMeta.Namespace,
				pod.ObjectMeta.Name, pod.ObjectMeta.UID,
				pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
				m.nodeName, node.ObjectMeta.UID,
				v.Name, sourceIndex,
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

		for sourceIndex, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			err := m.createPCR(
				ctx,
				pod.ObjectMeta.Namespace,
				pod.ObjectMeta.Name, pod.ObjectMeta.UID,
				pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
				m.nodeName, node.ObjectMeta.UID,
				v.Name, sourceIndex,
				source.PodCertificate.SignerName, source.PodCertificate.KeyType,
			)
			if err != nil {
				return fmt.Errorf("while creating pcr: %w", err)
			}
		}
	}

	return nil
}

// ForgetPod removes all live credential records associated with pod.
func (m *IssuingManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
	m.lock.Lock()
	defer m.lock.Unlock()

	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for sourceIndex, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			key := credentialKey{
				podUID:                     pod.ObjectMeta.UID,
				projectedVolumeName:        v.Name,
				projectedVolumeSourceIndex: sourceIndex,
			}
			delete(m.liveCredentials, key)
		}
	}
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

	movedToTerminalState := false
	for _, cond := range newPCR.Status.Conditions {
		switch cond.Type {
		case certificatesv1alpha1.PodCertificateRequestConditionTypeDenied,
			certificatesv1alpha1.PodCertificateRequestConditionTypeFailed,
			certificatesv1alpha1.PodCertificateRequestConditionTypeIssued:
			movedToTerminalState = true
		}
	}
	if !movedToTerminalState {
		return
	}

	m.lock.Lock()
	defer m.lock.Unlock()

	pendingCredential := m.pendingCredentials[newPCR.ObjectMeta.UID]
	if pendingCredential == nil {
		// If we don't have a pending credential record corresponding to this
		// PCR, there are two possible causes:
		//
		// 1) Some admin manually created a PCR that targets a pod on our node.
		//
		// 2) Kubelet created this PCR, but then restarted.
		//
		// In both cases, just do nothing.
		klog.InfoS("PodCertificateRequest update was for stray", "pcr", key)
		return
	}
	delete(m.pendingCredentials, newPCR.ObjectMeta.UID)

	credKey := credentialKey{
		podUID:                     newPCR.Spec.PodUID,
		projectedVolumeName:        pendingCredential.projectedVolumeName,
		projectedVolumeSourceIndex: pendingCredential.projectedVolumeSourceIndex,
	}
	m.liveCredentials[credKey] = &liveCredentialRecord{
		privateKey: pendingCredential.privateKey,
		pcr:        newPCR.DeepCopy(),
	}
}

func (m *IssuingManager) handlePCRDelete(obj any) {
	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = d.Obj
	}
	pcr := obj.(*certificatesv1alpha1.PodCertificateRequest)

	m.lock.Lock()
	defer m.lock.Unlock()
	delete(m.pendingCredentials, pcr.ObjectMeta.UID)
}

// createPCR creates a PodCertificateRequest and records it in the
// pendingCredentials map.  The PodCertificateRequest informer loop will then
// wait for it to be issued, denied, or failed, and update liveCredentials.
func (m *IssuingManager) createPCR(
	ctx context.Context,
	namespace string,
	podName string, podUID types.UID,
	serviceAccountName string, serviceAccountUID types.UID,
	nodeName types.NodeName, nodeUID types.UID,
	volumeName string, sourceIndex int, signerName, keyType string) error {

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

	m.pendingCredentials[req.ObjectMeta.UID] = &pendingCredentialRecord{
		privateKey:                 keyPEM,
		projectedVolumeName:        volumeName,
		projectedVolumeSourceIndex: sourceIndex,
	}

	return nil
}

func (m *IssuingManager) GetPodCertificateCredentialBundle(ctx context.Context, podUID types.UID, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	m.lock.Lock()
	defer m.lock.Unlock()

	key := credentialKey{
		podUID:                     podUID,
		projectedVolumeName:        volumeName,
		projectedVolumeSourceIndex: sourceIndex,
	}

	record := m.liveCredentials[key]
	if record == nil {
		return nil, nil, fmt.Errorf("no credentials yet")
	}

	for _, cond := range record.pcr.Status.Conditions {
		switch cond.Type {
		case certificatesv1alpha1.PodCertificateRequestConditionTypeDenied:
			return nil, nil, fmt.Errorf("PodCertificateRequest %s was denied", record.pcr.ObjectMeta.Namespace+"/"+record.pcr.ObjectMeta.Name)
		case certificatesv1alpha1.PodCertificateRequestConditionTypeFailed:
			return nil, nil, fmt.Errorf("PodCertificateRequest %s was failed", record.pcr.ObjectMeta.Namespace+"/"+record.pcr.ObjectMeta.Name)
		case certificatesv1alpha1.PodCertificateRequestConditionTypeIssued:
			return record.privateKey, []byte(record.pcr.Status.CertificateChain), nil
		}
	}
	return nil, nil, fmt.Errorf("no credentials yet; check status of PodCertificateRequest %s", record.pcr.ObjectMeta.Namespace+"/"+record.pcr.ObjectMeta.Name)
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

// NoOpManager is an implementation of Manager that just returns errors, meant
// for use in static/detached Kubelet mode.
type NoOpManager struct{}

var _ Manager = (*NoOpManager)(nil)

func (m *NoOpManager) TrackPod(ctx context.Context, pod *corev1.Pod) error {
	return nil
}

func (m *NoOpManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) GetPodCertificateCredentialBundle(ctx context.Context, podUID types.UID, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("unimplemented")
}
