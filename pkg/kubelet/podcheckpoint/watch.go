/*
Copyright The Kubernetes Authors.

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

package podcheckpoint

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	nodeinformers "k8s.io/client-go/informers/node/v1alpha1"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/parsers"
)

const podCheckpointWatchWorkers = 2

const podCheckpointAPITimeout = 10 * time.Second

var errPodCheckpointInFlight = errors.New("pod checkpoint is waiting for another pod operation")
var errPodCheckpointNotPending = errors.New("pod checkpoint was replaced or already completed")
var errPodCheckpointAPIUnavailable = errors.New("pod checkpoint is waiting for the API server")

func (m *manager) recordPodCheckpointEvent(ctx context.Context, namespace, name string, uid types.UID, eventType, reason, message string) {
	if m.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:       "PodCheckpoint",
		APIVersion: nodev1alpha1.SchemeGroupVersion.String(),
		Namespace:  namespace,
		Name:       name,
		UID:        uid,
	}
	m.recorder.WithLogger(klog.FromContext(ctx)).Event(ref, eventType, reason, message)
}

func (m *manager) recordPodCheckpointStarted(ctx context.Context, pod *v1.Pod, namespace, name string, uid types.UID) {
	m.recordPodCheckpointEvent(ctx, namespace, name, uid, v1.EventTypeNormal, events.PodCheckpointStarted, "Started checkpointing Pod")
	if m.recorder != nil {
		m.recorder.WithLogger(klog.FromContext(ctx)).Event(pod, v1.EventTypeNormal, events.SourcePodCheckpointing, "Checkpointing Pod")
	}
}

func podCheckpointTerminal(pc *nodev1alpha1.PodCheckpoint) bool {
	if pc.Status == nil {
		return false
	}
	cond := apimeta.FindStatusCondition(pc.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	return cond != nil && (cond.Status == metav1.ConditionTrue || cond.Reason == nodev1alpha1.PodCheckpointReasonFailed || cond.Reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced)
}

// podCheckpointWatchState holds the PodCheckpoint cache and queue of this
// node's checkpoint watch.
type podCheckpointWatchState struct {
	factory   informers.SharedInformerFactory
	indexer   cache.Indexer
	lister    nodelisters.PodCheckpointLister
	hasSynced cache.InformerSynced
	manager   *manager
	queue     workqueue.TypedRateLimitingInterface[string]
	// sweep requests removal of data whose PodCheckpoint was deleted.
	sweep chan struct{}
}

// pruneInterruptedCheckpoints forgets interrupted PodCheckpoints that were
// deleted before the watch could fail them.
func (m *manager) pruneInterruptedCheckpoints() {
	state := m.podCheckpointWatch
	if state == nil || state.hasSynced == nil || !state.hasSynced() {
		return
	}
	cached := sets.New[types.UID]()
	for _, object := range state.indexer.List() {
		if checkpoint, ok := object.(*nodev1alpha1.PodCheckpoint); ok {
			cached.Insert(checkpoint.UID)
		}
	}
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	for uid := range m.checkpointsInterrupted {
		if !cached.Has(uid) {
			m.checkpointsInterrupted.Delete(uid)
		}
	}
}

func (m *manager) runPodCheckpointWatch(ctx context.Context) {
	// Fail checkpoints interrupted by a restart before the informer starts, so
	// the watch does not capture them again.
	if !m.finalizeInterruptedCheckpoints(ctx) {
		return
	}
	m.startPodCheckpointWatch(ctx)
}

// newPodCheckpointWatch builds the watch of the PodCheckpoints assigned to this
// node. The API server filters it by status.nodeName so checkpoints for other
// nodes, including their captured Pod specs, never enter this cache. Run starts
// it.
func newPodCheckpointWatch(logger klog.Logger, m *manager) *podCheckpointWatchState {
	logger = klog.LoggerWithName(logger, "podcheckpoint-watch")
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{
			Name:   "podcheckpoint-watch",
			Logger: &logger,
		},
	)
	factory := informers.NewSharedInformerFactoryWithOptions(m.kubeClient, 0, informers.WithTweakListOptions(func(options *metav1.ListOptions) {
		options.FieldSelector = fields.OneTermEqualSelector("status.nodeName", string(m.nodeName)).String()
	}))
	checkpointInformer := factory.Node().V1alpha1().PodCheckpoints()
	informer := checkpointInformer.TypedInformer()
	state := &podCheckpointWatchState{factory: factory, indexer: informer.GetIndexer(), lister: checkpointInformer.Lister(), hasSynced: informer.HasSynced, queue: queue, manager: m, sweep: make(chan struct{}, 1)}
	if _, err := informer.AddTypedEventHandler(nodeinformers.PodCheckpointHandlerFuncs{AddFunc: state.enqueue, UpdateFunc: state.enqueueUpdate, DeleteFunc: state.enqueueDelete}, cache.HandlerOptions{Logger: &logger}); err != nil {
		// Registration fails only on an informer that has already stopped. The
		// kubelet keeps running without the watch, so the queue must not leak.
		logger.Error(err, "Failed to register PodCheckpoint event handler")
		queue.ShutDown()
		return nil
	}
	return state
}

// startPodCheckpointWatch executes checkpoints assigned to this node by the
// controller.
func (m *manager) startPodCheckpointWatch(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)
	// The cache sync helper reports its progress without naming the cache.
	logger := klog.LoggerWithName(klog.FromContext(ctx), "podcheckpoint-watch")
	ctx = klog.NewContext(ctx, logger)

	state := m.podCheckpointWatch
	if state == nil {
		logger.Info("Skipping PodCheckpoint watch: kubelet has no API client")
		return
	}

	var wg sync.WaitGroup
	// Workers must stop before the queue they read is shut down.
	defer func() {
		state.queue.ShutDown()
		wg.Wait()
	}()
	state.factory.StartWithContext(ctx)
	// Shutdown waits for the informer, which stops once ctx is canceled.
	defer state.factory.Shutdown()
	if !cache.WaitForNamedCacheSyncWithContext(ctx, state.hasSynced) {
		return
	}
	// Admission requires a Running source Pod, so once the Pod sources are
	// ready the Pod manager has every source Pod that still runs here.
	if err := wait.PollUntilContextCancel(ctx, time.Second, true, func(context.Context) (bool, error) {
		return m.sourcesReady.AllReady(), nil
	}); err != nil {
		return
	}
	m.pruneInterruptedCheckpoints()

	logger.Info("Starting PodCheckpoint watch")
	defer logger.Info("Shutting down PodCheckpoint watch")

	wg.Go(func() { state.runDataSweeper(ctx) })
	for range podCheckpointWatchWorkers {
		wg.Go(func() { wait.UntilWithContext(ctx, state.runWorker, time.Second) })
	}
	<-ctx.Done()
}

func (s *podCheckpointWatchState) enqueue(checkpoint *nodev1alpha1.PodCheckpoint) {
	s.queue.Add(cache.MetaObjectToName(checkpoint).String())
}

func (s *podCheckpointWatchState) enqueueUpdate(old, checkpoint *nodev1alpha1.PodCheckpoint) {
	if old.UID != checkpoint.UID {
		s.manager.pruneInterruptedCheckpoints()
		s.requestDataSweep()
	}
	s.enqueue(checkpoint)
}

func (s *podCheckpointWatchState) enqueueDelete(nodeinformers.DeletedPodCheckpoint) {
	s.manager.pruneInterruptedCheckpoints()
	s.requestDataSweep()
}

func (s *podCheckpointWatchState) requestDataSweep() {
	select {
	case s.sweep <- struct{}{}:
	default:
	}
}

// runDataSweeper removes data of deleted PodCheckpoints once the cache has
// synced and after each deletion. The periodic pass removes data that an
// operation or a restore still used when its PodCheckpoint was deleted.
func (s *podCheckpointWatchState) runDataSweeper(ctx context.Context) {
	logger := klog.FromContext(ctx)
	ticker := time.NewTicker(podCheckpointDataSweepPeriod)
	defer ticker.Stop()
	lastError := ""
	for {
		err := s.manager.sweepPodCheckpointData(ctx)
		if err == nil {
			lastError = ""
		} else if err.Error() != lastError {
			// Log a persistent removal failure once instead of on every pass.
			logger.Error(err, "Failed to remove data of deleted PodCheckpoints; retrying")
			lastError = err.Error()
		}
		select {
		case <-ctx.Done():
			return
		case <-s.sweep:
		case <-ticker.C:
		}
	}
}

func (s *podCheckpointWatchState) runWorker(ctx context.Context) {
	for s.processNext(ctx) {
	}
}

func (s *podCheckpointWatchState) processNext(ctx context.Context) bool {
	key, quit := s.queue.Get()
	if quit {
		return false
	}
	defer s.queue.Done(key)
	err := s.manager.syncPodCheckpoint(ctx, key)
	switch {
	case errors.Is(err, errPodCheckpointInFlight):
		s.queue.Forget(key)
		s.queue.AddAfter(key, time.Second)
	case err == nil || errors.Is(err, errPodCheckpointNotPending) || apierrors.IsNotFound(err):
		// A deleted PodCheckpoint leaves the cache. A new object with the same
		// name arrives as another watch event.
		s.queue.Forget(key)
	case apierrors.IsInvalid(err):
		if _, pending := s.manager.pendingPodCheckpointOutcome(key); pending {
			s.queue.AddRateLimited(key)
		} else {
			// Validation rejects the same status again; the next change to the
			// object queues it again.
			utilruntime.HandleErrorWithLogger(klog.FromContext(ctx), err, "PodCheckpoint status was rejected; waiting for the object to change", "key", key)
			s.queue.Forget(key)
		}
	default:
		// Forbidden is retried as well: the node authorizer allows the status
		// write only after it observes the assignment to this node.
		utilruntime.HandleErrorWithLogger(klog.FromContext(ctx), err, "Failed to sync PodCheckpoint", "key", key)
		s.queue.AddRateLimited(key)
	}
	return true
}

// syncPodCheckpoint reconciles a single PodCheckpoint. It is a no-op for objects
// in a terminal state. For a pending object whose source pod runs here it pins
// the instance by UID, records the in-progress status, and hands the checkpoint
// to the pod worker of the source pod, which captures it during SyncPod and
// finalizes the status asynchronously.
func (m *manager) syncPodCheckpoint(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	state := m.podCheckpointWatch
	if state == nil {
		return errors.New("PodCheckpoint informer is not initialized")
	}
	pc, err := state.lister.PodCheckpoints(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		if outcome, ok := m.pendingPodCheckpointOutcome(key); ok {
			return m.syncPendingPodCheckpointOutcome(ctx, key, outcome)
		}
		return nil
	}
	if err != nil {
		return err
	}
	if outcome, ok := m.pendingPodCheckpointOutcome(key); ok {
		return m.syncPendingPodCheckpointOutcome(ctx, key, outcome)
	}
	checkpointStatus := pc.Status
	if checkpointStatus == nil {
		checkpointStatus = &nodev1alpha1.PodCheckpointStatus{}
	}
	if podCheckpointTerminal(pc) {
		m.setCheckpointInterrupted(pc.UID, false)
		return nil
	}
	if checkpointStatus.NodeName == nil || *checkpointStatus.NodeName != string(m.nodeName) {
		return nil
	}
	if m.checkpointInterrupted(pc.UID) {
		err := m.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonFailed, "checkpoint failed: interrupted by a kubelet restart", nil, checkpointStatus.SourcePodUID)
		if err == nil || apierrors.IsNotFound(err) || errors.Is(err, errPodCheckpointNotPending) {
			m.setCheckpointInterrupted(pc.UID, false)
		}
		return err
	}

	sourcePodName := pc.Spec.SourcePod.Name
	pod, ok := m.podProvider.GetPodByName(namespace, sourcePodName)
	if !ok {
		// A capture of the deleted source fails when its Pod terminates.
		if checkpointStatus.SourcePodUID != nil && m.podCheckpointInFlight(*checkpointStatus.SourcePodUID) {
			return errPodCheckpointInFlight
		}
		// Admission requires a Running source Pod and the watch starts after the
		// Pod sources are ready, so the source Pod is no longer on this node.
		return m.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonFailed, fmt.Sprintf("source Pod %q is not running on node %q", sourcePodName, m.nodeName), nil, checkpointStatus.SourcePodUID)
	}

	// Pin the source pod instance by UID. A pod name can be reused, so a recorded
	// UID that no longer matches the live pod means the original instance was
	// replaced; fail rather than checkpoint the wrong instance.
	if uid := pc.Spec.SourcePod.UID; uid != nil && *uid != pod.UID {
		// Keep an existing claim; a first rejection identifies the observed replacement.
		sourcePodUID := checkpointStatus.SourcePodUID
		if sourcePodUID == nil {
			sourcePodUID = new(pod.UID)
		}
		return m.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q has UID %q but spec.sourcePod.uid pins it to %q; the original instance was replaced", sourcePodName, pod.UID, *uid),
			nil, sourcePodUID)
	}
	if checkpointStatus.SourcePodUID != nil && *checkpointStatus.SourcePodUID != pod.UID {
		return m.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q UID changed from %q to %q; the original instance was replaced", sourcePodName, *checkpointStatus.SourcePodUID, pod.UID),
			nil, checkpointStatus.SourcePodUID)
	}

	// A checkpoint of this pod is already pending or running; wait for it to finalize.
	if m.podCheckpointInFlight(pod.UID) {
		return errPodCheckpointInFlight
	}

	// Pin the source instance before execution. The pod worker captures the
	// template after reading runtime status so its image references describe
	// the same running containers selected for the CRI request.
	sourcePodUID := pod.UID
	if err := m.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
		nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress",
		nil, &sourcePodUID); err != nil {
		return err
	}

	// The kubelet enforces the checkpoint timeout with the deadline of the CRI
	// call, not a request field. The pod worker clamps it to the configured
	// ceiling (PodCheckpointTimeout); an unset TimeoutSeconds (zero timeout
	// here) falls back to that ceiling, so the operation always has a
	// deadline. Validation bounds a set value to [1, 3600] seconds.
	var timeout time.Duration
	if pc.Spec.TimeoutSeconds != nil {
		timeout = time.Duration(*pc.Spec.TimeoutSeconds) * time.Second
	}
	return m.requestPodCheckpoint(ctx, pod, &podCheckpointRequest{
		ctx:       ctx,
		namespace: namespace,
		name:      name,
		uid:       pc.UID,
		timeout:   timeout,
	})
}

// writePodCheckpointStatus sets the PodCheckpoint "Ready" condition and the
// execution status and captured pod template, preserving the controller
// assignment. Conflicts return to the watch queue or a later Pod sync. It
// publishes in-progress status and failures before capture; the runtime outcome
// is written by finalizePodCheckpointOutcome.
func (m *manager) writePodCheckpointStatus(ctx context.Context, namespace, name string, checkpointUID types.UID, status metav1.ConditionStatus, reason, message string, template *v1.PodTemplateSpec, sourcePodUID *types.UID) error {
	if m.kubeClient == nil {
		return errors.New("cannot update PodCheckpoint: kubelet has no API client")
	}
	ctx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	failed := reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced || reason == nodev1alpha1.PodCheckpointReasonFailed
	pc, err := m.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	// The informer may lag a completion or a delete/recreate. Do not let an
	// old queue entry overwrite the result or start another runtime operation.
	if pc.UID != checkpointUID || podCheckpointTerminal(pc) {
		return errPodCheckpointNotPending
	}
	if pc.Status == nil {
		pc.Status = &nodev1alpha1.PodCheckpointStatus{}
	}

	if pc.Status.NodeName != nil && *pc.Status.NodeName != string(m.nodeName) {
		return errPodCheckpointNotPending
	}

	// Another reconciliation may have claimed the source while this work was queued.
	if sourcePodUID != nil && pc.Status.SourcePodUID != nil && *sourcePodUID != *pc.Status.SourcePodUID {
		return errPodCheckpointNotPending
	}

	apimeta.SetStatusCondition(&pc.Status.Conditions, metav1.Condition{
		Type:               nodev1alpha1.PodCheckpointConditionReady,
		Status:             status,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: pc.Generation,
	})
	pc.Status.NodeName = new(string(m.nodeName))
	if sourcePodUID != nil {
		pc.Status.SourcePodUID = sourcePodUID
	}
	if template != nil {
		pc.Status.CheckpointedPodTemplate = template
		pc.Status.CheckpointedContainers = checkpointedContainerStatuses(&template.Spec)
	}
	if failed {
		pc.Status.CompletionTime = new(metav1.Now())
	}

	_, err = m.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).UpdateStatus(ctx, pc, metav1.UpdateOptions{})
	if err == nil && failed && pc.DeletionTimestamp == nil {
		m.recordPodCheckpointEvent(ctx, namespace, name, checkpointUID, v1.EventTypeWarning, events.CheckpointFailed, "Checkpoint failed. See PodCheckpoint status for details")
	}
	return err
}

// publishPodCheckpointTemplate captures the Pod template and writes it to the
// PodCheckpoint status with the source Pod UID and the in-progress Ready
// condition. It uses the runtime snapshot that selected the CRI container IDs.
// Resolving the spec's tag again could select an image different from the one
// running in those containers.
func (m *manager) publishPodCheckpointTemplate(ctx context.Context, namespace, name string, checkpointUID types.UID, pod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
	template := checkpointutil.SanitizePodTemplate(pod)
	pinImage := func(container *v1.Container) error {
		status := podStatus.FindContainerStatusByName(container.Name)
		if status == nil || status.State != kubecontainer.ContainerStateRunning {
			return fmt.Errorf("cannot capture checkpoint template: container %q is not running", container.Name)
		}
		repository, _, digest, err := parsers.ParseImageName(status.ImageRef)
		if err != nil || digest == "" {
			return fmt.Errorf("cannot capture checkpoint template: runtime must report a repository digest reference for container %q, got %q", container.Name, status.ImageRef)
		}
		container.Image = repository + "@" + digest
		return nil
	}
	for container := range kubecontainer.CheckpointableContainers(&template.Spec) {
		if err := pinImage(container); err != nil {
			return err
		}
	}
	err := m.writePodCheckpointStatus(ctx, namespace, name, checkpointUID, metav1.ConditionFalse,
		nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", template, &pod.UID)
	if apierrors.IsNotFound(err) {
		// The PodCheckpoint was deleted after the request.
		return fmt.Errorf("%w: %w", errPodCheckpointNotPending, err)
	}
	if apierrors.IsConflict(err) || isRetriableAPIError(err) {
		return fmt.Errorf("%w: capture checkpoint template: %w", errPodCheckpointAPIUnavailable, err)
	}
	return err
}

// checkpointedContainerStatuses builds the convenience status list of the
// containers captured in a checkpoint, in the order checkpointPodContainerIDs
// selects them. Container names are unique within a pod, so a single list
// covers restartable init and regular containers.
func checkpointedContainerStatuses(spec *v1.PodSpec) []nodev1alpha1.PodCheckpointContainerStatus {
	out := make([]nodev1alpha1.PodCheckpointContainerStatus, 0, len(spec.InitContainers)+len(spec.Containers))
	for container := range kubecontainer.CheckpointableContainers(spec) {
		out = append(out, nodev1alpha1.PodCheckpointContainerStatus{Name: container.Name})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// isRetriableAPIError reports whether err is a temporary API failure, after
// which the kubelet retries the status write instead of failing the checkpoint.
func isRetriableAPIError(err error) bool {
	if apierrors.IsNotFound(err) || apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err) {
		return false
	}
	if apierrors.IsTimeout(err) || apierrors.IsServerTimeout(err) ||
		apierrors.IsServiceUnavailable(err) || apierrors.IsTooManyRequests(err) ||
		apierrors.IsInternalError(err) || errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, context.Canceled) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var networkError net.Error
	return errors.As(err, &networkError)
}
