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
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1alpha1 "k8s.io/client-go/informers/certificates/v1alpha1"
	coreinformersv1 "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// PodManager is a local wrapper interface for pod.Manager.
type PodManager interface {
	GetPodByName(namespace, name string) (*corev1.Pod, bool)
	GetPods() []*corev1.Pod
}

// Manager abstracts the functionality needed by Kubelet and the volume host in
// order to provide pod certificate functionality.
type Manager interface {
	// TrackPod is called by Kubelet every time a new pod is assigned to the node.
	TrackPod(ctx context.Context, pod *corev1.Pod)
	// ForgetPod is called by Kubelet every time a pod is dropped from the node.
	ForgetPod(ctx context.Context, pod *corev1.Pod)

	// GetPodCertificateCredentialBundle is called by the volume host to
	// retrieve the credential bundle for a given pod certificate volume.
	GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, volumeName string, sourceIndex int) ([]byte, []byte, error)
}

// IssuingManager is the main implementation of Manager.
//
// The core construct is a workqueue that contains one entry for each
// PodCertificateProjection (tracked with a podname/volumename/sourceindex
// tuple) in each non-mirror Pod scheduled to the node. Entries are held in the
// workqueue until the PodCertificateProjection has a corresponding non-stale
// entry in liveCredentials.
//
// Note that the entry in liveCredentials doesn't imply that a certificate was
// successfully issued.  PodCertificateRequests that end with Denied or Failed
// conditions.
//
// Refresh is handled by periodicially redriving all PodCertificateProjections
// into the queue.
type IssuingManager struct {
	kc kubernetes.Interface

	podManager PodManager

	projectionQueue workqueue.TypedRateLimitingInterface[projectionKey]

	pcrInformer cache.SharedIndexInformer
	pcrLister   certlistersv1alpha1.PodCertificateRequestLister

	nodeInformer cache.SharedIndexInformer
	nodeLister   corelistersv1.NodeLister
	nodeName     types.NodeName

	clock clock.PassiveClock

	// lock covers credStore
	lock      sync.Mutex
	credStore map[projectionKey]*projectionRecord
}

type projectionKey struct {
	namespace   string
	podName     string
	volumeName  string
	sourceIndex int
}

type projectionRecord struct {
	// lock covers all fields within projectionRecord.
	lock sync.Mutex

	// The state machine for this projection.  Moves through the following
	// states:
	//
	//                            ┌─────────────────┐
	//                            ▼                 │
	//  initial ────► wait ────► fresh ──────► waitrefresh
	//                            │                 │
	//                            ├───► denied ◄────┤
	//                            │                 │
	//                            └───► failed ◄────┘
	curState credState
}

// Interface type for all projection record states.
type credState interface {
	getCredBundle() (privKey, certChain []byte, err error)
}

type credStateInitial struct {
}

func (c *credStateInitial) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("credential bundle is not issued yet")
}

type credStateWait struct {
	privateKey []byte
	pcrName    string
}

func (c *credStateWait) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("credential bundle is not issued yet")
}

type credStateDenied struct {
	Reason  string
	Message string
}

func (c *credStateDenied) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("PodCertificateRequest was permanently denied: reason=%q message=%q", c.Reason, c.Message)
}

type credStateFailed struct {
	Reason  string
	Message string
}

func (c *credStateFailed) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("PodCertificateRequest was permanently failed: reason=%q message=%q", c.Reason, c.Message)
}

type credStateFresh struct {
	privateKey     []byte
	certChain      []byte
	beginRefreshAt time.Time
}

func (c *credStateFresh) getCredBundle() ([]byte, []byte, error) {
	return c.privateKey, c.certChain, nil
}

type credStateWaitRefresh struct {
	privateKey []byte
	certChain  []byte

	refreshPrivateKey []byte
	refreshPCRName    string
}

func (c *credStateWaitRefresh) getCredBundle() ([]byte, []byte, error) {
	return c.privateKey, c.certChain, nil
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, podManager PodManager, pcrInformer certinformersv1alpha1.PodCertificateRequestInformer, nodeInformer coreinformersv1.NodeInformer, nodeName types.NodeName, clock clock.WithTicker) *IssuingManager {
	m := &IssuingManager{
		kc: kc,

		podManager:      podManager,
		projectionQueue: workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[projectionKey]()),

		pcrInformer:  pcrInformer.Informer(),
		pcrLister:    pcrInformer.Lister(),
		nodeInformer: nodeInformer.Informer(),
		nodeLister:   nodeInformer.Lister(),
		nodeName:     nodeName,
		clock:        clock,

		credStore: map[projectionKey]*projectionRecord{},
	}

	// Add informer functions for PodCertificateRequests.  In all cases, we just
	// queue the corresponding PodCertificateProjections for re-processing.
	// This is not needed for correctness, since volumeSourceQueue backoffs will
	// eventually trigger the volume to be inspected.  However, it's a better UX
	// for us to notice immediately once the certificate is issued.
	m.pcrInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			pcr := obj.(*certificatesv1alpha1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.ObjectMeta.Namespace, pcr.Spec.PodName)
		},
		UpdateFunc: func(old, new any) {
			pcr := new.(*certificatesv1alpha1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.ObjectMeta.Namespace, pcr.Spec.PodName)
		},
		DeleteFunc: func(obj any) {
			pcr := obj.(*certificatesv1alpha1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.ObjectMeta.Namespace, pcr.Spec.PodName)
		},
	})

	return m
}

func (m *IssuingManager) queueAllProjectionsForPod(namespace, name string) {
	pod, ok := m.podManager.GetPodByName(namespace, name)
	if !ok {
		return
	}

	forEachPodCertificateProjection(pod, func(volName string, srcIndex int, _ *corev1.PodCertificateProjection) error {
		m.projectionQueue.Add(projectionKey{
			namespace:   namespace,
			podName:     name,
			volumeName:  volName,
			sourceIndex: srcIndex,
		})
		return nil
	})
}

func (m *IssuingManager) Run(ctx context.Context) {
	klog.InfoS("podcertificate.IssuingManager starting up")
	if !cache.WaitForCacheSync(ctx.Done(), m.pcrInformer.HasSynced, m.nodeInformer.HasSynced) {
		return
	}
	go wait.JitterUntilWithContext(ctx, m.runRefreshPass, 1*time.Minute, 1.0, false)
	go wait.UntilWithContext(ctx, m.runProjectionProcessor, time.Second)
	<-ctx.Done()

	m.projectionQueue.ShutDown()

	klog.InfoS("podcertificate.IssuingManager shut down")
}

func (m *IssuingManager) runProjectionProcessor(ctx context.Context) {
	for m.processNextProjection(ctx) {
	}
}

func (m *IssuingManager) processNextProjection(ctx context.Context) bool {
	key, quit := m.projectionQueue.Get()
	if quit {
		return false
	}
	defer m.projectionQueue.Done(key)

	err := m.handleProjection(ctx, key)
	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "while handling podCertificate projected volume source", "namespace", key.namespace, "pod", key.podName, "volume", key.volumeName, "sourceIndex", key.sourceIndex)
		m.projectionQueue.AddRateLimited(key)
		return true
	}

	m.projectionQueue.Forget(key)
	return true
}

func (m *IssuingManager) handleProjection(ctx context.Context, key projectionKey) error {
	// Remember, returning nil from this function indicates that the work item
	// was successfully processed, and should be dropped from the queue.  We
	// should not return nil until liveCredentials contains a non-stale entry
	// for the projection.

	pod, ok := m.podManager.GetPodByName(key.namespace, key.podName)
	if !ok {
		// If we can't find the pod anymore, it's been deleted.  Clear all our
		// internal state associated with the pod and return a nil error so it
		// is forgotten from the queue.

		m.lock.Lock()
		defer m.lock.Unlock()
		for k := range m.credStore {
			if k.namespace == key.namespace && k.podName == key.podName {
				delete(m.credStore, k)
			}
		}

		return nil
	}

	var source *corev1.PodCertificateProjection
	for _, vol := range pod.Spec.Volumes {
		if vol.Name == key.volumeName && vol.Projected != nil {
			for i, volumeSource := range vol.Projected.Sources {
				if i == key.sourceIndex && volumeSource.PodCertificate != nil {
					source = volumeSource.PodCertificate
				}
			}
		}
	}
	if source == nil {
		return fmt.Errorf("podCertificate projected volume source %s/%s/%s/%d doesn't exist", key.namespace, key.podName, key.volumeName, key.sourceIndex)
	}

	// We fetch the service account so we can know its UID.  Ideally, Kubelet
	// would have a central component that tracks all service accounts related
	// to pods on the node using a single-item watch.
	serviceAccount, err := m.kc.CoreV1().ServiceAccounts(pod.ObjectMeta.Namespace).Get(ctx, pod.Spec.ServiceAccountName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("while fetching service account: %w", err)
	}

	node, err := m.nodeLister.Get(string(m.nodeName))
	if err != nil {
		return fmt.Errorf("while getting node object from local cache: %w", err)
	}

	var rec *projectionRecord
	func() {
		m.lock.Lock()
		defer m.lock.Unlock()

		credKey := projectionKey{
			namespace:   key.namespace,
			podName:     key.podName,
			volumeName:  key.volumeName,
			sourceIndex: key.sourceIndex,
		}
		rec = m.credStore[credKey]

		if rec == nil {
			rec = &projectionRecord{
				curState: &credStateInitial{},
			}
			m.credStore[credKey] = rec
		}
	}()

	// Lock the record for the remainder of the function.
	rec.lock.Lock()
	defer rec.lock.Unlock()

	switch state := rec.curState.(type) {
	case *credStateInitial:
		// We have not started the initial issuance.  We need to create a PCR
		// and record it in credStore.
		privKey, pcr, err := m.createPodCertificateRequest(
			ctx,
			pod.ObjectMeta.Namespace,
			pod.ObjectMeta.Name, pod.ObjectMeta.UID,
			pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
			m.nodeName, node.ObjectMeta.UID,
			source.SignerName, source.KeyType, source.MaxExpirationSeconds,
		)
		if err != nil {
			return fmt.Errorf("while creating initial PodCertificateRequest: %w", err)
		}

		rec.curState = &credStateWait{
			privateKey: privKey,
			pcrName:    pcr.ObjectMeta.Name,
		}

		return fmt.Errorf("created PodCertificateRequest; must wait for it to be issued")

	case *credStateWait:
		// We are working through the initial issuance.  We created a PCR, now
		// we need to wait for it to reach a terminal state.
		pcr, err := m.pcrLister.PodCertificateRequests(key.namespace).Get(state.pcrName)
		if err != nil {
			return fmt.Errorf("while getting PodCertificateRequest %q: %w", key.namespace+"/"+state.pcrName, err)
		}

		// TODO(KEP-4317): "Not Found" could be due to either informer lag, or
		// because someone deleted the pending PCR.

		// If the PodCertificateRequest has moved to a terminal state, update
		// our state machine accordingly.
		for _, cond := range pcr.Status.Conditions {
			switch cond.Type {
			case certificatesv1alpha1.PodCertificateRequestConditionTypeDenied:
				rec.curState = &credStateDenied{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				return nil
			case certificatesv1alpha1.PodCertificateRequestConditionTypeFailed:
				rec.curState = &credStateFailed{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				return nil
			case certificatesv1alpha1.PodCertificateRequestConditionTypeIssued:
				rec.curState = &credStateFresh{
					privateKey:     state.privateKey,
					certChain:      []byte(pcr.Status.CertificateChain),
					beginRefreshAt: pcr.Status.BeginRefreshAt.Time,
				}
				return nil
			}
		}

		// Nothing -- the request is still pending.  Retain this
		// PodCertificateProjection in the workqueue.
		return fmt.Errorf("the PodCertificateRequest is still pending")

	case *credStateDenied:
		// Nothing to do; this is a permanent error state for the pod.
		return nil

	case *credStateFailed:
		// Nothing to do; this is a permanent error state for the pod.
		return nil

	case *credStateFresh:
		if m.clock.Now().Before(state.beginRefreshAt) {
			// If it's not time to refresh yet, do nothing.  clea
			return nil
		}

		privKey, pcr, err := m.createPodCertificateRequest(
			ctx,
			pod.ObjectMeta.Namespace,
			pod.ObjectMeta.Name, pod.ObjectMeta.UID,
			pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
			m.nodeName, node.ObjectMeta.UID,
			source.SignerName, source.KeyType, source.MaxExpirationSeconds,
		)
		if err != nil {
			return fmt.Errorf("while creating refresh PodCertificateRequest: %w", err)
		}

		rec.curState = &credStateWaitRefresh{
			privateKey: state.privateKey,
			certChain:  state.certChain,

			refreshPrivateKey: privKey,
			refreshPCRName:    pcr.ObjectMeta.Name,
		}

		return fmt.Errorf("created refresh PodCertificateRequest; must wait for it to be issued")

	case *credStateWaitRefresh:
		// Check the refresh PodCertificateRequest
		pcr, err := m.pcrLister.PodCertificateRequests(key.namespace).Get(state.refreshPCRName)
		if err != nil {
			return fmt.Errorf("while getting PodCertificateRequest %q: %w", key.namespace+"/"+state.refreshPCRName, err)
		}

		// TODO(KEP-4317): "Not Found" could be due to either informer lag, or
		// because someone deleted the pending PCR.

		// If the PodCertificateRequest has moved to a terminal state, update
		// our state machine accordingly.
		for _, cond := range pcr.Status.Conditions {
			switch cond.Type {
			case certificatesv1alpha1.PodCertificateRequestConditionTypeDenied:
				rec.curState = &credStateDenied{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				return nil
			case certificatesv1alpha1.PodCertificateRequestConditionTypeFailed:
				rec.curState = &credStateFailed{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				return nil
			case certificatesv1alpha1.PodCertificateRequestConditionTypeIssued:
				rec.curState = &credStateFresh{
					privateKey:     state.refreshPrivateKey,
					certChain:      []byte(pcr.Status.CertificateChain),
					beginRefreshAt: pcr.Status.BeginRefreshAt.Time,
				}
				return nil
			}
		}

		// Nothing -- the request is still pending.  Retain this
		// PodCertificateProjection in the workqueue.
		return fmt.Errorf("the refresh PodCertificateRequest is still pending")
	}

	return nil
}

func forEachPodCertificateProjection(pod *corev1.Pod, fn func(volName string, sourceIndex int, source *corev1.PodCertificateProjection) error) []error {
	var errs []error
	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for sourceIndex, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			if err := fn(v.Name, sourceIndex, source.PodCertificate); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errs
}

// runRefreshPass adds every non-mirror pod on the node back to the volume
// source processing queue.
func (m *IssuingManager) runRefreshPass(ctx context.Context) {
	allPods := m.podManager.GetPods()
	for _, pod := range allPods {
		m.queueAllProjectionsForPod(pod.ObjectMeta.Namespace, pod.ObjectMeta.Name)
	}
}

// TrackPod queues the pod's podCertificate projected volume sources for
// processing.
func (m *IssuingManager) TrackPod(ctx context.Context, pod *corev1.Pod) {
	m.queueAllProjectionsForPod(pod.ObjectMeta.Namespace, pod.ObjectMeta.Name)
}

// ForgetPod queues the pod's podCertificate projected volume sources for processing.
//
// The pod worker will notice that the pod no longer exists and clear any
// pending and live credentials associated with it.
func (m *IssuingManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
	m.queueAllProjectionsForPod(pod.ObjectMeta.Namespace, pod.ObjectMeta.Name)
}

// createPodCertificateRequest creates a PodCertificateRequest.
func (m *IssuingManager) createPodCertificateRequest(
	ctx context.Context,
	namespace string,
	podName string, podUID types.UID,
	serviceAccountName string, serviceAccountUID types.UID,
	nodeName types.NodeName, nodeUID types.UID,
	signerName, keyType string, maxExpirationSeconds *int32) ([]byte, *certificatesv1alpha1.PodCertificateRequest, error) {

	privateKey, publicKey, proof, err := m.generateKeyAndProof(keyType, []byte(podUID))
	if err != nil {
		return nil, nil, fmt.Errorf("while generating keypair: %w", err)
	}

	pkixPublicKey, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return nil, nil, fmt.Errorf("while marshaling public key: %w", err)
	}

	keyPEM, err := pemEncodeKey(privateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("while PEM-encoding private key: %w", err)
	}

	req := &certificatesv1alpha1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    namespace,
			GenerateName: "req-",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "core/v1",
					Kind:       "Pod",
					Name:       podName,
					UID:        podUID,
				},
			},
		},
		Spec: certificatesv1alpha1.PodCertificateRequestSpec{
			SignerName:           signerName,
			PodName:              podName,
			PodUID:               podUID,
			ServiceAccountName:   serviceAccountName,
			ServiceAccountUID:    serviceAccountUID,
			NodeName:             nodeName,
			NodeUID:              nodeUID,
			MaxExpirationSeconds: maxExpirationSeconds,
			PKIXPublicKey:        pkixPublicKey,
			ProofOfPossession:    proof,
		},
	}

	req, err = m.kc.CertificatesV1alpha1().PodCertificateRequests(namespace).Create(ctx, req, metav1.CreateOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("while creating on API: %w", err)
	}

	return keyPEM, req, err
}

func (m *IssuingManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	var rec *projectionRecord
	func() {
		m.lock.Lock()
		defer m.lock.Unlock()

		credKey := projectionKey{
			namespace:   namespace,
			podName:     podName,
			volumeName:  volumeName,
			sourceIndex: sourceIndex,
		}
		rec = m.credStore[credKey]

	}()

	if rec == nil {
		return nil, nil, fmt.Errorf("no credentials yet")
	}

	rec.lock.Lock()
	defer rec.lock.Unlock()

	return rec.curState.getCredBundle()
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

func (m *NoOpManager) TrackPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("unimplemented")
}
