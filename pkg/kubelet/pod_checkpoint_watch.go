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

package kubelet

import (
	"context"
	"errors"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	nodeinformers "k8s.io/client-go/informers/node/v1alpha1"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/parsers"
)

const podCheckpointWatchWorkers = 2

const podCheckpointAPITimeout = 10 * time.Second

const podCheckpointSourceIndex = "sourcePod"

var errPodCheckpointInFlight = errors.New("pod checkpoint is waiting for another pod operation")
var errPodCheckpointNotPending = errors.New("pod checkpoint was replaced or already completed")
var errPodCheckpointAPIUnavailable = errors.New("pod checkpoint is waiting for the API server")
var errPodCheckpointPolicyUnavailable = errors.New("pod checkpoint is waiting for RuntimeClass option policy")

func (kl *Kubelet) recordPodCheckpointEvent(ctx context.Context, namespace, name string, uid types.UID, eventType, reason, message string) {
	if kl.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		Kind:       "PodCheckpoint",
		APIVersion: nodev1alpha1.SchemeGroupVersion.String(),
		Namespace:  namespace,
		Name:       name,
		UID:        uid,
	}
	kl.recorder.WithLogger(klog.FromContext(ctx)).Event(ref, eventType, reason, message)
}

func (kl *Kubelet) recordPodCheckpointStarted(ctx context.Context, pod *v1.Pod, namespace, name string, uid types.UID) {
	kl.recordPodCheckpointEvent(ctx, namespace, name, uid, v1.EventTypeNormal, events.CheckpointStarted, "Started checkpointing Pod")
	if kl.recorder != nil {
		kl.recorder.WithLogger(klog.FromContext(ctx)).Event(pod, v1.EventTypeNormal, events.CheckpointingPod, "Checkpointing Pod")
	}
}

func podCheckpointTerminal(pc *nodev1alpha1.PodCheckpoint) bool {
	cond := apimeta.FindStatusCondition(pc.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	return cond != nil && (cond.Status == metav1.ConditionTrue || cond.Reason == nodev1alpha1.PodCheckpointReasonFailed || cond.Reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced)
}

// podCheckpointWatchState connects local Pod observation with the checkpoint
// watch so either event ordering eventually schedules the checkpoint.
type podCheckpointWatchState struct {
	indexer   cache.Indexer
	lister    nodelisters.PodCheckpointLister
	hasSynced cache.InformerSynced
	kubelet   *Kubelet
	queue     workqueue.TypedRateLimitingInterface[string]
}

// cachedPodCheckpoint returns the checkpoint from this node's PodCheckpoint
// watch, or nil when the watch has not synced or does not hold it.
func (kl *Kubelet) cachedPodCheckpoint(namespace, name string) *nodev1alpha1.PodCheckpoint {
	state := kl.podCheckpointWatch.Load()
	if state == nil || state.hasSynced == nil || !state.hasSynced() {
		return nil
	}
	pc, err := state.lister.PodCheckpoints(namespace).Get(name)
	if err != nil {
		return nil
	}
	return pc
}

func (kl *Kubelet) prunePodCheckpointMarkers() {
	state := kl.podCheckpointWatch.Load()
	if state == nil || state.hasSynced == nil || !state.hasSynced() {
		return
	}

	// Collect markers before the cache snapshot so a newly arriving checkpoint
	// cannot lose protection because it was absent from an older snapshot.
	unused := make(map[types.UID]struct{})
	collect := func(key, _ interface{}) bool {
		unused[key.(types.UID)] = struct{}{}
		return true
	}
	kl.checkpointRecoveryPending.Range(collect)
	kl.checkpointCleanupBlocked.Range(collect)
	if len(unused) == 0 {
		return
	}
	for _, object := range state.indexer.List() {
		if checkpoint, ok := object.(*nodev1alpha1.PodCheckpoint); ok {
			delete(unused, checkpoint.UID)
		}
	}
	if kl.podCheckpointWatch.Load() != state {
		return
	}
	for uid := range unused {
		kl.checkpointRecoveryPending.Delete(uid)
		kl.checkpointCleanupBlocked.Delete(uid)
	}
}

func (kl *Kubelet) enqueuePodCheckpoints(pod *v1.Pod) {
	state := kl.podCheckpointWatch.Load()
	if state == nil {
		return
	}
	checkpoints, err := state.indexer.ByIndex(podCheckpointSourceIndex, pod.Namespace+"/"+pod.Name)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to find checkpoints for Pod %s/%s: %w", pod.Namespace, pod.Name, err))
		return
	}
	for _, checkpoint := range checkpoints {
		key, err := cache.MetaNamespaceKeyFunc(checkpoint)
		if err == nil {
			state.queue.Add(key)
		}
	}
}

// Index by both namespace and name so Pod additions only visit checkpoints
// referencing that Pod, regardless of how many checkpoints the cluster retains.
func podCheckpointSourceIndexFunc(obj interface{}) ([]string, error) {
	checkpoint, ok := obj.(*nodev1alpha1.PodCheckpoint)
	if !ok {
		return nil, fmt.Errorf("expected PodCheckpoint, got %T", obj)
	}
	if checkpoint.Spec.SourcePod == nil || checkpoint.Spec.SourcePod.Name == "" {
		return nil, nil
	}
	return []string{checkpoint.Namespace + "/" + checkpoint.Spec.SourcePod.Name}, nil
}

func (kl *Kubelet) runPodCheckpointWatch(ctx context.Context) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) {
		return
	}
	// Complete recovery before starting the informer. Otherwise an interrupted
	// object can be observed and restarted while recovery is still marking it failed.
	if !kl.finalizeInterruptedCheckpoints(ctx) {
		return
	}
	kl.startPodCheckpointWatch(ctx)
}

// startPodCheckpointWatch executes checkpoints assigned to this node by the
// controller. The API server filters the watch by status.nodeName so checkpoints
// for other nodes, including their captured Pod specs, never enter this cache.
func (kl *Kubelet) startPodCheckpointWatch(ctx context.Context) {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)

	if kl.kubeClient == nil {
		logger.Info("Skipping PodCheckpoint watch: kubelet has no API client")
		return
	}

	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
	defer queue.ShutDown()

	informer := nodeinformers.NewFilteredPodCheckpointInformer(kl.kubeClient, metav1.NamespaceAll, 0, cache.Indexers{
		cache.NamespaceIndex:     cache.MetaNamespaceIndexFunc,
		podCheckpointSourceIndex: podCheckpointSourceIndexFunc,
	}, func(options *metav1.ListOptions) {
		options.FieldSelector = fields.OneTermEqualSelector("status.nodeName", string(kl.nodeName)).String()
	})
	state := &podCheckpointWatchState{indexer: informer.GetIndexer(), lister: nodelisters.NewPodCheckpointLister(informer.GetIndexer()), hasSynced: informer.HasSynced, queue: queue, kubelet: kl}
	if _, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{AddFunc: state.enqueue, UpdateFunc: state.enqueueUpdate, DeleteFunc: state.enqueueDelete}); err != nil {
		logger.Error(err, "Failed to register PodCheckpoint event handler")
		return
	}
	kl.podCheckpointWatch.Store(state)
	defer kl.podCheckpointWatch.Store(nil)

	go informer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), informer.HasSynced) {
		logger.Error(nil, "Failed to sync PodCheckpoint informer cache")
		return
	}
	kl.prunePodCheckpointMarkers()

	logger.Info("Starting PodCheckpoint watch")
	defer logger.Info("Shutting down PodCheckpoint watch")

	for range podCheckpointWatchWorkers {
		go wait.UntilWithContext(ctx, state.runWorker, time.Second)
	}
	<-ctx.Done()
}

func (s *podCheckpointWatchState) enqueue(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	s.queue.Add(key)
}

func (s *podCheckpointWatchState) enqueueUpdate(old, obj interface{}) {
	oldCheckpoint, oldOK := old.(*nodev1alpha1.PodCheckpoint)
	checkpoint, ok := obj.(*nodev1alpha1.PodCheckpoint)
	if oldOK && ok && oldCheckpoint.UID != checkpoint.UID {
		s.kubelet.prunePodCheckpointMarkers()
	}
	s.enqueue(obj)
}

func (s *podCheckpointWatchState) enqueueDelete(_ interface{}) {
	s.kubelet.prunePodCheckpointMarkers()
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
	err := s.kubelet.syncPodCheckpoint(ctx, key)
	if errors.Is(err, errPodCheckpointInFlight) {
		s.queue.Forget(key)
		s.queue.AddAfter(key, time.Second)
	} else if err != nil && !errors.Is(err, errPodCheckpointNotPending) {
		utilruntime.HandleError(fmt.Errorf("PodCheckpoint %q sync failed: %w", key, err))
		s.queue.AddRateLimited(key)
	} else {
		s.queue.Forget(key)
	}
	return true
}

// syncPodCheckpoint reconciles a single PodCheckpoint. It is a no-op for objects
// in a terminal state and for objects whose source pod is not managed by this
// kubelet (another node's kubelet handles those). For a pending object whose
// source pod runs here it pins the instance by UID, records the in-progress
// status, and hands the checkpoint to the pod worker of the source pod, which
// captures it during SyncPod and finalizes the status asynchronously.
func (kl *Kubelet) syncPodCheckpoint(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	state := kl.podCheckpointWatch.Load()
	if state == nil {
		return errors.New("PodCheckpoint informer is not initialized")
	}
	pc, err := state.lister.PodCheckpoints(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if _, recovering := kl.checkpointRecoveryPending.Load(pc.UID); recovering {
		return kl.finalizeRecoveredPodCheckpoint(ctx, pc)
	}
	if _, blocked := kl.checkpointCleanupBlocked.Load(pc.UID); blocked {
		// Startup recovery could not safely remove the deterministic output for
		// this object. Leave it in progress so cleanup can be retried on the next
		// kubelet start instead of overwriting or orphaning the existing data.
		return nil
	}

	// Skip objects that have reached a terminal state.
	if podCheckpointTerminal(pc) {
		return nil
	}

	// A checkpoint can arrive before the Pod manager observes its source.
	// Local Pod additions requeue matching checkpoints from the informer store.
	if pc.Spec.SourcePod == nil || pc.Spec.SourcePod.Name == "" {
		// Validation rejects an unset source pod reference; nothing to act on.
		return nil
	}
	sourcePodName := pc.Spec.SourcePod.Name
	pod, ok := kl.podManager.GetPodByName(namespace, sourcePodName)
	if !ok {
		return nil
	}

	// Pin the source pod instance by UID. A pod name can be reused, so a recorded
	// UID that no longer matches the live pod means the original instance was
	// replaced; fail rather than checkpoint the wrong instance.
	if uid := pc.Spec.SourcePod.UID; uid != nil && *uid != pod.UID {
		// Keep an existing claim; a first rejection identifies the observed replacement.
		sourcePodUID := pc.Status.SourcePodUID
		if sourcePodUID == nil {
			sourcePodUID = new(pod.UID)
		}
		return kl.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q has UID %q but spec.sourcePod.uid pins it to %q; the original instance was replaced", sourcePodName, pod.UID, *uid),
			nil, sourcePodUID)
	}
	if pc.Status.SourcePodUID != nil && *pc.Status.SourcePodUID != pod.UID {
		return kl.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q UID changed from %q to %q; the original instance was replaced", sourcePodName, *pc.Status.SourcePodUID, pod.UID),
			nil, pc.Status.SourcePodUID)
	}

	// A checkpoint of this pod is already pending or running; wait for it to finalize.
	if kl.podCheckpointInFlight(pod.UID) {
		return errPodCheckpointInFlight
	}

	// Pin the source instance before execution. The pod worker captures the
	// template after reading runtime status so its image references describe
	// the same running containers selected for the CRI request.
	sourcePodUID := pod.UID
	if err := kl.writePodCheckpointStatus(ctx, namespace, name, pc.UID, metav1.ConditionFalse,
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
	return kl.requestPodCheckpoint(ctx, pod, &podCheckpointRequest{
		ctx:       ctx,
		namespace: namespace,
		name:      name,
		uid:       pc.UID,
		timeout:   timeout,
		options:   pc.Spec.CheckpointOptions,
	})
}

// writePodCheckpointStatus sets the PodCheckpoint "Ready" condition and the
// execution status and captured pod template, preserving the controller
// assignment through conflict-retried status updates. It handles in-progress
// and SourcePodReplaced transitions; the terminal outcome is written by
// finalizePodCheckpoint.
func (kl *Kubelet) writePodCheckpointStatus(ctx context.Context, namespace, name string, checkpointUID types.UID, status metav1.ConditionStatus, reason, message string, template *v1.PodTemplateSpec, sourcePodUID *types.UID) error {
	if kl.kubeClient == nil {
		return errors.New("cannot update PodCheckpoint: kubelet has no API client")
	}
	ctx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	recordFailure := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pc, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		// The informer may lag a completion or a delete/recreate. Do not let an
		// old queue entry overwrite the result or start another runtime operation.
		if pc.UID != checkpointUID || podCheckpointTerminal(pc) {
			return errPodCheckpointNotPending
		}

		if pc.Status.NodeName != nil && *pc.Status.NodeName != string(kl.nodeName) {
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
		pc.Status.NodeName = new(string(kl.nodeName))
		if sourcePodUID != nil {
			pc.Status.SourcePodUID = sourcePodUID
		}
		if template != nil {
			pc.Status.CheckpointedPodTemplate = template
			pc.Status.CheckpointedContainers = checkpointedContainerStatuses(&template.Spec)
		}

		_, err = kl.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).UpdateStatus(ctx, pc, metav1.UpdateOptions{})
		if err == nil && reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced && pc.DeletionTimestamp == nil {
			recordFailure = true
		}
		return err
	})
	if err == nil && recordFailure {
		kl.recordPodCheckpointEvent(ctx, namespace, name, checkpointUID, v1.EventTypeWarning, events.CheckpointFailed, "Checkpoint failed. See PodCheckpoint status for details")
	}
	return err
}

// publishPodCheckpointTemplate captures the Pod template and writes it to the
// PodCheckpoint status with the source Pod UID and the in-progress Ready
// condition. It uses the runtime snapshot that selected the CRI container IDs.
// Resolving the spec's tag again could select an image different from the one
// running in those containers.
func (kl *Kubelet) publishPodCheckpointTemplate(ctx context.Context, namespace, name string, checkpointUID types.UID, pod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
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
	for container := range kubecontainer.CheckpointContainers(&template.Spec) {
		if err := pinImage(container); err != nil {
			return err
		}
	}
	err := kl.writePodCheckpointStatus(ctx, namespace, name, checkpointUID, metav1.ConditionFalse,
		nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", template, &pod.UID)
	if apierrors.IsNotFound(err) {
		// The PodCheckpoint was deleted after the request.
		return fmt.Errorf("%w: %w", errPodCheckpointNotPending, err)
	}
	if checkpointutil.RuntimeOptionPolicyUnavailable(err) {
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
	for container := range kubecontainer.CheckpointContainers(spec) {
		out = append(out, nodev1alpha1.PodCheckpointContainerStatus{Name: container.Name})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
