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
	"bytes"
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
	mathrand "math/rand/v2"
	"sync"
	"time"

	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1beta1 "k8s.io/client-go/informers/certificates/v1beta1"
	coreinformersv1 "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	certlistersv1beta1 "k8s.io/client-go/listers/certificates/v1beta1"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// PodManager is a local wrapper interface for pod.Manager.
type PodManager interface {
	GetPodByUID(uid types.UID) (*corev1.Pod, bool)
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
	GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, podUID, volumeName string, sourceIndex int) (privKey []byte, certChain []byte, err error)

	// MetricReport returns a snapshot of current pod certificate states for this manager.
	MetricReport() *MetricReport
}

// MetricReport contains metrics about the current state of pod certificate projected volume sources.
type MetricReport struct {
	PodCertificateStates map[SignerAndState]int
}

// SignerAndState represents a combination of a signer name and the state of a pod certificate.
type SignerAndState struct {
	SignerName string
	State      string
}

// After this amount of time (plus jitter), we can assume that a PCR that we
// created, but isn't showing up on our informer, must have been deleted.
const assumeDeletedThreshold = 10 * time.Minute

// After this amount of time since the BeginRefreshAt of the certificate, we
// consider a certificate to be overdue for refresh.
const refreshOverdueDuration = 10 * time.Minute

// IssuingManager is the main implementation of Manager.
//
// The core construct is a workqueue that contains one entry for each
// PodCertificateProjection (tracked with a podname/volumename/sourceindex
// tuple) in each non-mirror Pod scheduled to the node. Everytime anything
// interesting happens to a PodCertificateRequest or Pod, we redrive all of the
// potentially-affected PodCertificateProjections into the workqueue.
//
// State is not preserved across restarts --- if Kubelet or the node restarts,
// then all PodCertificateProjections will be queued for immediate refresh.
//
// Refresh is handled by periodicially redriving all PodCertificateProjections
// into the queue.
type IssuingManager struct {
	kc kubernetes.Interface

	podManager PodManager

	recorder record.EventRecorder

	projectionQueue workqueue.TypedRateLimitingInterface[projectionKey]

	pcrInformer cache.SharedIndexInformer
	pcrLister   certlistersv1beta1.PodCertificateRequestLister

	nodeInformer cache.SharedIndexInformer
	nodeLister   corelistersv1.NodeLister
	nodeName     types.NodeName

	clock clock.PassiveClock

	// lock covers credStore
	lock      sync.Mutex
	credStore map[projectionKey]*projectionRecord
}

type projectionKey struct {
	Namespace   string
	PodName     string
	PodUID      string
	VolumeName  string
	SourceIndex int
}

type projectionRecord struct {
	// lock covers all fields within projectionRecord.
	lock sync.Mutex

	// The state machine for this projection:
	//
	//
	//                         ┌─────────────────┐
	//                         ▼                 │
	// fresh ────► wait ────► fresh ──────► waitrefresh
	//               │                           │
	//               ├──────► denied ◄───────────┤
	//               │                           │
	//               └──────► failed ◄───────────┘
	curState credState
}

// Interface type for all projection record states.
type credState interface {
	getCredBundle() (privKey, certChain []byte, err error)
	metricsState(now time.Time) string
}

type credStateInitial struct {
}

func (c *credStateInitial) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("credential bundle is not issued yet")
}

func (c *credStateInitial) metricsState(_ time.Time) string {
	return "not_yet_issued"
}

type credStateWait struct {
	privateKey []byte
	pcrName    string
	// If it has reached this time and the PCR isn't showing up on the informer,
	// assume that it was deleted.
	pcrAbandonAt time.Time
}

func (c *credStateWait) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("credential bundle is not issued yet")
}

func (c *credStateWait) metricsState(_ time.Time) string {
	return "not_yet_issued"
}

type credStateDenied struct {
	Reason  string
	Message string
}

func (c *credStateDenied) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("PodCertificateRequest was permanently denied: reason=%q message=%q", c.Reason, c.Message)
}

func (c *credStateDenied) metricsState(_ time.Time) string {
	return "denied"
}

type credStateFailed struct {
	Reason  string
	Message string
}

func (c *credStateFailed) getCredBundle() ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("PodCertificateRequest was permanently failed: reason=%q message=%q", c.Reason, c.Message)
}

func (c *credStateFailed) metricsState(_ time.Time) string {
	return "failed"
}

type credStateFresh struct {
	privateKey                       []byte
	certChain                        []byte
	beginRefreshAt                   time.Time
	notAfter                         time.Time
	eventEmittedForOverdueForRefresh bool
	eventEmittedForExpiration        bool
}

func (c *credStateFresh) getCredBundle() ([]byte, []byte, error) {
	return c.privateKey, c.certChain, nil
}

func (c *credStateFresh) metricsState(now time.Time) string {
	if now.After(c.notAfter) {
		return "expired"
	}
	if now.After(c.beginRefreshAt.Add(refreshOverdueDuration)) {
		return "overdue_for_refresh"
	}
	return "fresh"
}

type credStateWaitRefresh struct {
	privateKey                       []byte
	certChain                        []byte
	beginRefreshAt                   time.Time
	notAfter                         time.Time
	eventEmittedForOverdueForRefresh bool
	eventEmittedForExpiration        bool

	refreshPrivateKey []byte
	refreshPCRName    string
	// If it has reached this time and the PCR isn't showing up on the informer,
	// assume that it was deleted.
	refreshPCRAbandonAt time.Time
}

func (c *credStateWaitRefresh) getCredBundle() ([]byte, []byte, error) {
	return c.privateKey, c.certChain, nil
}

func (c *credStateWaitRefresh) metricsState(now time.Time) string {
	if now.After(c.notAfter) {
		return "expired"
	}
	if now.After(c.beginRefreshAt.Add(refreshOverdueDuration)) {
		return "overdue_for_refresh"
	}
	return "fresh"
}

var _ Manager = (*IssuingManager)(nil)

func NewIssuingManager(kc kubernetes.Interface, podManager PodManager, recorder record.EventRecorder, pcrInformer certinformersv1beta1.PodCertificateRequestInformer, nodeInformer coreinformersv1.NodeInformer, nodeName types.NodeName, clock clock.WithTicker) *IssuingManager {
	m := &IssuingManager{
		kc: kc,

		podManager:      podManager,
		recorder:        recorder,
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
			pcr := obj.(*certificatesv1beta1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.Spec.PodUID)
		},
		UpdateFunc: func(old, new any) {
			pcr := new.(*certificatesv1beta1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.Spec.PodUID)
		},
		DeleteFunc: func(obj any) {
			pcr := obj.(*certificatesv1beta1.PodCertificateRequest)
			m.queueAllProjectionsForPod(pcr.Spec.PodUID)
		},
	})

	return m
}

func (m *IssuingManager) queueAllProjectionsForPod(uid types.UID) {
	pod, ok := m.podManager.GetPodByUID(uid)
	if !ok {
		return
	}

	for _, v := range pod.Spec.Volumes {
		if v.Projected == nil {
			continue
		}

		for sourceIndex, source := range v.Projected.Sources {
			if source.PodCertificate == nil {
				continue
			}

			key := projectionKey{
				Namespace:   pod.ObjectMeta.Namespace,
				PodName:     pod.ObjectMeta.Name,
				PodUID:      string(pod.ObjectMeta.UID),
				VolumeName:  v.Name,
				SourceIndex: sourceIndex,
			}
			m.projectionQueue.Add(key)
		}
	}
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
		utilruntime.HandleErrorWithContext(ctx, err, "while handling podCertificate projected volume source", "namespace", key.Namespace, "pod", key.PodName, "volume", key.VolumeName, "sourceIndex", key.SourceIndex)
		m.projectionQueue.AddRateLimited(key)
		return true
	}

	m.projectionQueue.Forget(key)
	return true
}

func (m *IssuingManager) handleProjection(ctx context.Context, key projectionKey) error {
	// Remember, returning nil from this function indicates that the work item
	// was successfully processed, and should be dropped from the queue.

	pod, ok := m.podManager.GetPodByUID(types.UID(key.PodUID))
	if !ok {
		// If we can't find the pod anymore, it's been deleted.  Clear all our
		// internal state associated with the pod and return a nil error so it
		// is forgotten from the queue.
		m.cleanupCredStoreForPod(key.Namespace, key.PodName, key.PodUID)

		return nil
	}

	var source *corev1.PodCertificateProjection
	for _, vol := range pod.Spec.Volumes {
		if vol.Name == key.VolumeName && vol.Projected != nil {
			for i, volumeSource := range vol.Projected.Sources {
				if i == key.SourceIndex && volumeSource.PodCertificate != nil {
					source = volumeSource.PodCertificate
				}
			}
		}
	}
	if source == nil {
		// No amount of retrying will fix this problem.  Log it and return nil.
		utilruntime.HandleErrorWithContext(ctx, nil, "pod does not contain the named podCertificate projected volume source", "key", key)
		return nil
	}

	var rec *projectionRecord
	func() {
		m.lock.Lock()
		defer m.lock.Unlock()

		rec = m.credStore[key]

		if rec == nil {
			rec = &projectionRecord{
				curState: &credStateInitial{},
			}
			m.credStore[key] = rec
		}
	}()

	// Lock the record for the remainder of the function.
	rec.lock.Lock()
	defer rec.lock.Unlock()

	switch state := rec.curState.(type) {
	case *credStateInitial:
		// We have not started the initial issuance.  We need to create a PCR
		// and record it in credStore.

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

		privKey, pcr, err := m.createPodCertificateRequest(
			ctx,
			pod.ObjectMeta.Namespace,
			pod.ObjectMeta.Name, pod.ObjectMeta.UID,
			pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
			m.nodeName, node.ObjectMeta.UID,
			source.SignerName, source.KeyType, source.MaxExpirationSeconds, source.UserAnnotations,
		)
		if err != nil {
			return fmt.Errorf("while creating initial PodCertificateRequest: %w", err)
		}

		rec.curState = &credStateWait{
			privateKey:   privKey,
			pcrName:      pcr.ObjectMeta.Name,
			pcrAbandonAt: pcr.ObjectMeta.CreationTimestamp.Time.Add(assumeDeletedThreshold + jitterDuration()),
		}

		// Return nil to remove the projection from the workqueue --- it will be
		// readded once the PodCertificateRequest appears in the informer cache,
		// and goes through status updates.
		klog.V(4).InfoS("PodCertificateRequest created, moving to credStateWait", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
		return nil

	case *credStateWait:
		// We are working through the initial issuance.  We created a PCR, now
		// we need to wait for it to reach a terminal state.

		pcr, err := m.pcrLister.PodCertificateRequests(key.Namespace).Get(state.pcrName)
		if k8serrors.IsNotFound(err) && m.clock.Now().After(state.pcrAbandonAt) {
			// "Not Found" could be due to informer lag, or because someone
			// deleted the PodCertificateRequest.  In the first case, the
			// correct action is to continue waiting.  In the second case, the
			// correct action is to recreate the PCR.  Properly disambiguating
			// the cases will require resourceVersions to be ordered, and for
			// the lister to report the highest resource version it has seen. In
			// the meantime, assume that if it has been 10 minutes since we
			// remember creating the PCR, then we must be in case 2.  Return to
			// credStateInitial so we create a new PCR.
			rec.curState = &credStateInitial{}
			return fmt.Errorf("PodCertificateRequest %q appears to have been deleted", key.Namespace+"/"+state.pcrName)
		} else if err != nil {
			return fmt.Errorf("while getting PodCertificateRequest %q: %w", key.Namespace+"/"+state.pcrName, err)
		}

		// If the PodCertificateRequest has moved to a terminal state, update
		// our state machine accordingly.
		for _, cond := range pcr.Status.Conditions {
			switch cond.Type {
			case certificatesv1beta1.PodCertificateRequestConditionTypeDenied:
				rec.curState = &credStateDenied{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				klog.V(4).InfoS("PodCertificateRequest denied, moving to credStateDenied", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				eventMessage := fmt.Sprintf("PodCertificateRequest %s was denied, reason=%q, message=%q", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name, cond.Reason, cond.Message)
				m.recorder.Eventf(pod, corev1.EventTypeWarning, certificatesv1beta1.PodCertificateRequestConditionTypeDenied, cond.Reason, eventMessage)
				return nil
			case certificatesv1beta1.PodCertificateRequestConditionTypeFailed:
				rec.curState = &credStateFailed{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				klog.V(4).InfoS("PodCertificateRequest failed, moving to credStateFailed", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				eventMessage := fmt.Sprintf("PodCertificateRequest %s failed, reason=%q, message=%q", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name, cond.Reason, cond.Message)
				m.recorder.Eventf(pod, corev1.EventTypeWarning, certificatesv1beta1.PodCertificateRequestConditionTypeFailed, cond.Reason, eventMessage)
				return nil
			case certificatesv1beta1.PodCertificateRequestConditionTypeIssued:
				rec.curState = &credStateFresh{
					privateKey:     state.privateKey,
					certChain:      cleanCertificateChain([]byte(pcr.Status.CertificateChain)),
					beginRefreshAt: pcr.Status.BeginRefreshAt.Time.Add(jitterDuration()),
					notAfter:       pcr.Status.NotAfter.Time,
				}
				klog.V(4).InfoS("PodCertificateRequest issued, moving to credStateFresh", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				return nil
			}
		}

		// Nothing -- the request is still pending.  Return nil to remove the
		// projection from the workqueue.  It will be redriven when the
		// PodCertificateRequest gets an update.
		klog.V(4).InfoS("PodCertificateRequest not in terminal state, remaining in credStateWait", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
		return nil

	case *credStateDenied:
		// Nothing to do; this is a permanent error state for the pod.
		klog.V(4).InfoS("staying in credStateDenied", "key", key)
		return nil

	case *credStateFailed:
		// Nothing to do; this is a permanent error state for the pod.
		klog.V(4).InfoS("staying in credStateFailed", "key", key)
		return nil

	case *credStateFresh:
		// Do nothing until it is time to refresh, then create a new PCR and
		// switch to credStateWaitRefresh.

		if m.clock.Now().Before(state.beginRefreshAt) {
			// If it's not time to refresh yet, do nothing.
			return nil
		}

		klog.V(4).InfoS("Time to refresh", "key", key)

		// The current time is more than 10 minutes past the most recently issued certificate's `beginRefreshAt` timestamp but the state has not been labeled with overdue for refresh.
		if m.clock.Now().After(state.beginRefreshAt.Add(refreshOverdueDuration)) && !state.eventEmittedForOverdueForRefresh {
			klog.V(4).InfoS("Refresh overdue", "key", key)
			m.recorder.Eventf(pod, corev1.EventTypeWarning, "CertificateOverdueForRefresh", "PodCertificate refresh overdue")
			state.eventEmittedForOverdueForRefresh = true
		}

		// The current time is past the most recently issued certificate's `notAfter` timestamp but the state has not been labelled with expired.
		if m.clock.Now().After(state.notAfter) && !state.eventEmittedForExpiration {
			klog.V(4).InfoS("Certificates expired", "key", key)
			m.recorder.Eventf(pod, corev1.EventTypeWarning, "CertificateExpired", "PodCertificate expired")
			state.eventEmittedForExpiration = true
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

		privKey, pcr, err := m.createPodCertificateRequest(
			ctx,
			pod.ObjectMeta.Namespace,
			pod.ObjectMeta.Name, pod.ObjectMeta.UID,
			pod.Spec.ServiceAccountName, serviceAccount.ObjectMeta.UID,
			m.nodeName, node.ObjectMeta.UID,
			source.SignerName, source.KeyType, source.MaxExpirationSeconds, source.UserAnnotations,
		)
		if err != nil {
			return fmt.Errorf("while creating refresh PodCertificateRequest: %w", err)
		}

		rec.curState = &credStateWaitRefresh{
			privateKey:                       state.privateKey,
			certChain:                        state.certChain,
			beginRefreshAt:                   state.beginRefreshAt,
			notAfter:                         state.notAfter,
			eventEmittedForOverdueForRefresh: state.eventEmittedForOverdueForRefresh,
			eventEmittedForExpiration:        state.eventEmittedForExpiration,

			refreshPrivateKey:   privKey,
			refreshPCRName:      pcr.ObjectMeta.Name,
			refreshPCRAbandonAt: pcr.ObjectMeta.CreationTimestamp.Time.Add(assumeDeletedThreshold + jitterDuration()),
		}

		// Return nil to remove the projection from the workqueue --- it will be
		// readded once the PodCertificateRequest appears in the informer cache,
		// and goes through status updates.
		klog.V(4).InfoS("PodCertificateRequest created, moving to credStateWaitRefresh", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
		return nil

	case *credStateWaitRefresh:
		// Check the refresh PodCertificateRequest
		pcr, err := m.pcrLister.PodCertificateRequests(key.Namespace).Get(state.refreshPCRName)
		if k8serrors.IsNotFound(err) && m.clock.Now().After(state.refreshPCRAbandonAt) {
			// "Not Found" could be due to informer lag, or because someone
			// deleted the PodCertificateRequest.  In the first case, the
			// correct action is to continue waiting.  In the second case, the
			// correct action is to recreate the PCR.  Properly disambiguating
			// the cases will require resourceVersions to be ordered, and for
			// the lister to report the highest resource version it has seen. In
			// the meantime, assume that if it has been 10 minutes since we
			// remember creating the PCR, then we must be in case 2.  Return to
			// credStateFresh so we create a new PCR.
			rec.curState = &credStateFresh{
				privateKey:                       state.privateKey,
				certChain:                        state.certChain,
				beginRefreshAt:                   state.beginRefreshAt,
				notAfter:                         state.notAfter,
				eventEmittedForOverdueForRefresh: state.eventEmittedForOverdueForRefresh,
				eventEmittedForExpiration:        state.eventEmittedForExpiration,
			}
			return fmt.Errorf("PodCertificateRequest appears to have been deleted")
		} else if err != nil {
			return fmt.Errorf("while getting PodCertificateRequest %q: %w", key.Namespace+"/"+state.refreshPCRName, err)
		}

		// If the PodCertificateRequest has moved to a terminal state, update
		// our state machine accordingly.
		for _, cond := range pcr.Status.Conditions {
			switch cond.Type {
			case certificatesv1beta1.PodCertificateRequestConditionTypeDenied:
				rec.curState = &credStateDenied{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				klog.V(4).InfoS("PodCertificateRequest denied, moving to credStateDenied", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				eventMessage := fmt.Sprintf("PodCertificateRequest %s was denied, reason=%q, message=%q", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name, cond.Reason, cond.Message)
				m.recorder.Eventf(pod, corev1.EventTypeWarning, certificatesv1beta1.PodCertificateRequestConditionTypeDenied, cond.Reason, eventMessage)
				return nil
			case certificatesv1beta1.PodCertificateRequestConditionTypeFailed:
				rec.curState = &credStateFailed{
					Reason:  cond.Reason,
					Message: cond.Message,
				}
				klog.V(4).InfoS("PodCertificateRequest failed, moving to credStateFailed", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				eventMessage := fmt.Sprintf("PodCertificateRequest %s failed, reason=%q, message=%q", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name, cond.Reason, cond.Message)
				m.recorder.Eventf(pod, corev1.EventTypeWarning, certificatesv1beta1.PodCertificateRequestConditionTypeFailed, cond.Reason, eventMessage)
				return nil
			case certificatesv1beta1.PodCertificateRequestConditionTypeIssued:
				rec.curState = &credStateFresh{
					privateKey:     state.refreshPrivateKey,
					certChain:      cleanCertificateChain([]byte(pcr.Status.CertificateChain)),
					beginRefreshAt: pcr.Status.BeginRefreshAt.Time.Add(jitterDuration()),
					notAfter:       pcr.Status.NotAfter.Time,
				}
				klog.V(4).InfoS("PodCertificateRequest issued, moving to credStateFresh", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
				return nil
			}
		}

		// Nothing -- the request is still pending.  Return nil to remove the
		// projection from the workqueue.  It will be redriven when the
		// PodCertificateRequest gets an update.
		klog.V(4).InfoS("PodCertificateRequest not in terminal state, remaining in credStateWaitRefresh", "key", key, "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)

		// The current time is more than 10 minutes past the most recently issued certificate's `beginRefreshAt` timestamp but the state has not been labeled with overdue for refresh.
		if m.clock.Now().After(state.beginRefreshAt.Add(refreshOverdueDuration)) && !state.eventEmittedForOverdueForRefresh {
			klog.V(4).InfoS("Refresh overdue", "key", key)
			m.recorder.Eventf(pod, corev1.EventTypeWarning, "CertificateOverdueForRefresh", "PodCertificate refresh overdue")
			state.eventEmittedForOverdueForRefresh = true
		}

		// The current time is past the most recently issued certificate's `notAfter` timestamp but the state has not been labeled with expired.
		if m.clock.Now().After(state.notAfter) && !state.eventEmittedForExpiration {
			klog.V(4).InfoS("Certificates expired", "key", key)
			m.recorder.Eventf(pod, corev1.EventTypeWarning, "CertificateExpired", "PodCertificate expired")
			state.eventEmittedForExpiration = true
		}
		return nil
	}

	return nil
}

// jitterDuration returns a 5-minute randomized jitter to the given duration, to
// prevent multiple PodCertificateProjections from synchronizing their PCR
// creations.
func jitterDuration() time.Duration {
	return time.Duration(mathrand.Int64N(5 * 60 * 1_000_000_000))
}

// runRefreshPass adds every non-mirror pod on the node back to the volume
// source processing queue.
func (m *IssuingManager) runRefreshPass(ctx context.Context) {
	allPods := m.podManager.GetPods()
	for _, pod := range allPods {
		m.queueAllProjectionsForPod(pod.ObjectMeta.UID)
	}
}

// TrackPod queues the pod's podCertificate projected volume sources for
// processing.
func (m *IssuingManager) TrackPod(ctx context.Context, pod *corev1.Pod) {
	m.queueAllProjectionsForPod(pod.ObjectMeta.UID)
}

// cleanupCredStoreForPod removes all credStore entries for the specified pod.
func (m *IssuingManager) cleanupCredStoreForPod(namespace, podName, podUID string) {
	m.lock.Lock()
	defer m.lock.Unlock()

	for k := range m.credStore {
		if k.Namespace == namespace && k.PodName == podName && k.PodUID == podUID {
			delete(m.credStore, k)
		}
	}
}

// ForgetPod cleans up all pod certificate credentials for the specified pod.
//
// The pod worker will notice that the pod no longer exists and clear any
// pending and live credentials associated with it.
func (m *IssuingManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
	// Immediately clean up credStore entries for this pod to prevent race conditions
	m.cleanupCredStoreForPod(pod.Namespace, pod.Name, string(pod.UID))
}

// createPodCertificateRequest creates a PodCertificateRequest.
func (m *IssuingManager) createPodCertificateRequest(
	ctx context.Context,
	namespace string,
	podName string, podUID types.UID,
	serviceAccountName string, serviceAccountUID types.UID,
	nodeName types.NodeName, nodeUID types.UID,
	signerName, keyType string, maxExpirationSeconds *int32, userAnnotations map[string]string) ([]byte, *certificatesv1beta1.PodCertificateRequest, error) {

	privateKey, publicKey, proof, err := generateKeyAndProof(keyType, []byte(podUID))
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

	req := &certificatesv1beta1.PodCertificateRequest{
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
		Spec: certificatesv1beta1.PodCertificateRequestSpec{
			SignerName:                signerName,
			PodName:                   podName,
			PodUID:                    podUID,
			ServiceAccountName:        serviceAccountName,
			ServiceAccountUID:         serviceAccountUID,
			NodeName:                  nodeName,
			NodeUID:                   nodeUID,
			MaxExpirationSeconds:      maxExpirationSeconds,
			PKIXPublicKey:             pkixPublicKey,
			ProofOfPossession:         proof,
			UnverifiedUserAnnotations: userAnnotations,
		},
	}

	req, err = m.kc.CertificatesV1beta1().PodCertificateRequests(namespace).Create(ctx, req, metav1.CreateOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("while creating on API: %w", err)
	}

	return keyPEM, req, nil
}

func (m *IssuingManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, podUID, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	credKey := projectionKey{
		Namespace:   namespace,
		PodName:     podName,
		PodUID:      podUID,
		VolumeName:  volumeName,
		SourceIndex: sourceIndex,
	}

	var rec *projectionRecord
	func() {
		m.lock.Lock()
		defer m.lock.Unlock()
		rec = m.credStore[credKey]
	}()

	if rec == nil {
		return nil, nil, fmt.Errorf("no credentials yet for key=%v", credKey)
	}

	rec.lock.Lock()
	defer rec.lock.Unlock()

	return rec.curState.getCredBundle()
}

func (m *IssuingManager) MetricReport() *MetricReport {
	report := &MetricReport{
		PodCertificateStates: map[SignerAndState]int{},
	}

	// Iterate through all pods and their podCertificate projected volume sources
	// instead of iterating through credStore, so that we can use the SignerName
	// of the podCertificate projection source.
	allPods := m.podManager.GetPods()
	for _, pod := range allPods {
		for _, v := range pod.Spec.Volumes {
			if v.Projected == nil {
				continue
			}

			for sourceIndex, source := range v.Projected.Sources {
				if source.PodCertificate == nil {
					continue
				}

				key := projectionKey{
					Namespace:   pod.ObjectMeta.Namespace,
					PodName:     pod.ObjectMeta.Name,
					PodUID:      string(pod.ObjectMeta.UID),
					VolumeName:  v.Name,
					SourceIndex: sourceIndex,
				}

				var rec *projectionRecord
				func() {
					m.lock.Lock()
					defer m.lock.Unlock()
					rec = m.credStore[key]
				}()
				if rec == nil {
					continue
				}

				func() {
					rec.lock.Lock()
					defer rec.lock.Unlock()

					metricsKey := SignerAndState{
						SignerName: source.PodCertificate.SignerName,
						State:      rec.curState.metricsState(m.clock.Now()),
					}
					report.PodCertificateStates[metricsKey]++
				}()

			}
		}
	}

	return report
}

func hashBytes(in []byte) []byte {
	out := sha256.Sum256(in)
	return out[:]
}

func generateKeyAndProof(keyType string, toBeSigned []byte) (privKey crypto.PrivateKey, pubKey crypto.PublicKey, sig []byte, err error) {
	switch keyType {
	case "RSA3072":
		key, err := rsa.GenerateKey(rand.Reader, 3072)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 3072 key: %w", err)
		}
		sig, err := rsa.SignPSS(rand.Reader, key, crypto.SHA256, hashBytes(toBeSigned), nil)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while signing proof: %w", err)
		}
		return key, &key.PublicKey, sig, nil
	case "RSA4096":
		key, err := rsa.GenerateKey(rand.Reader, 4096)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating RSA 4096 key: %w", err)
		}
		sig, err := rsa.SignPSS(rand.Reader, key, crypto.SHA256, hashBytes(toBeSigned), nil)
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
	case "ECDSAP521":
		key, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("while generating ECDSA P521 key: %w", err)
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

// ensure that all inter-block data and block headers are dropped from the
// certificate chain.
func cleanCertificateChain(in []byte) []byte {
	outChain := &bytes.Buffer{}

	rest := in
	var b *pem.Block
	for {
		b, rest = pem.Decode(rest)
		if b == nil {
			break
		}

		cleanedBlock := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: b.Bytes,
		}
		outChain.Write(pem.EncodeToMemory(cleanedBlock))
	}

	return outChain.Bytes()
}

// NoOpManager is an implementation of Manager that just returns errors, meant
// for use in static/detached Kubelet mode.
type NoOpManager struct{}

var _ Manager = (*NoOpManager)(nil)

func (m *NoOpManager) TrackPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {
}

func (m *NoOpManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, podUID, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	return nil, nil, fmt.Errorf("unimplemented")
}

func (m *NoOpManager) MetricReport() *MetricReport {
	return &MetricReport{}
}
