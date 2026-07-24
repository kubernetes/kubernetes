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

	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	checkpointutil "k8s.io/kubernetes/pkg/apis/checkpoint/util"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const podCheckpointWatchWorkers = 2

var (
	errPodCheckpointInFlight       = errors.New("pod checkpoint is waiting for another pod operation")
	errPodCheckpointPodNotObserved = errors.New("source pod is assigned to this node but is not yet observed by the pod manager")
)

// startPodCheckpointWatch runs a node-side watch on PodCheckpoint objects and
// executes checkpoints for the pods this kubelet runs (KEP-5823). The kubelet,
// not a control-plane controller, is the component that performs a checkpoint:
// it observes PodCheckpoint objects and acts on those whose source pod
// (spec.sourcePodName) it manages, so no control-plane-to-kubelet call is
// needed. For alpha the watch is cluster-wide and filtered locally by pod
// ownership; PodCheckpoint objects are low-volume and short-lived, so this is
// acceptable. Narrowing the watch to this node with a field selector is a
// non-breaking follow-up. It runs until ctx is cancelled.
func (kl *Kubelet) startPodCheckpointWatch(ctx context.Context) {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)

	if kl.dynamicClient == nil {
		logger.Info("Skipping PodCheckpoint watch: kubelet has no API client")
		return
	}

	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
	defer queue.ShutDown()

	enqueue := func(obj interface{}) {
		key, err := cache.MetaNamespaceKeyFunc(obj)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		queue.Add(key)
	}

	listWatcher := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (apiruntime.Object, error) {
			return kl.dynamicClient.Resource(podCheckpointGVR).Namespace(metav1.NamespaceAll).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return kl.dynamicClient.Resource(podCheckpointGVR).Namespace(metav1.NamespaceAll).Watch(ctx, options)
		},
	}

	_, informer := cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: listWatcher,
		ObjectType:    &unstructured.Unstructured{},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc:    enqueue,
			UpdateFunc: func(_, newObj interface{}) { enqueue(newObj) },
		},
	})

	go informer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), informer.HasSynced) {
		logger.Error(nil, "Failed to sync PodCheckpoint informer cache")
		return
	}

	logger.Info("Starting PodCheckpoint watch")
	defer logger.Info("Shutting down PodCheckpoint watch")

	processNext := func() bool {
		key, quit := queue.Get()
		if quit {
			return false
		}
		defer queue.Done(key)
		if err := kl.syncPodCheckpoint(ctx, key); errors.Is(err, errPodCheckpointInFlight) || errors.Is(err, errPodCheckpointPodNotObserved) {
			queue.Forget(key)
			queue.AddAfter(key, time.Second)
			return true
		} else if err != nil {
			utilruntime.HandleError(fmt.Errorf("PodCheckpoint %q sync failed: %w", key, err))
			queue.AddRateLimited(key)
			return true
		}
		queue.Forget(key)
		return true
	}
	worker := func(context.Context) {
		for processNext() {
		}
	}

	for range podCheckpointWatchWorkers {
		go wait.UntilWithContext(ctx, worker, time.Second)
	}
	<-ctx.Done()
}

// syncPodCheckpoint reconciles a single PodCheckpoint. It is a no-op for objects
// in a terminal state and for objects whose source pod is not managed by this
// kubelet (another node's kubelet handles those). For a pending object whose
// source pod runs here it pins the instance by UID, records the in-progress
// status and the captured pod template, and starts the checkpoint via the CRI
// path (Kubelet.CheckpointPod), which finalizes the status asynchronously.
func (kl *Kubelet) syncPodCheckpoint(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	obj, err := kl.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	var pc checkpointv1alpha1.PodCheckpoint
	if err := apiruntime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pc); err != nil {
		return fmt.Errorf("failed to read PodCheckpoint %q: %w", name, err)
	}
	if _, blocked := kl.checkpointCleanupBlocked.Load(pc.UID); blocked {
		// Startup recovery could not safely remove the deterministic output for
		// this object. Leave it in progress so cleanup can be retried on the next
		// kubelet start instead of overwriting or orphaning the existing data.
		return nil
	}

	// Skip objects that have reached a terminal state.
	if cond := apimeta.FindStatusCondition(pc.Status.Conditions, checkpointv1alpha1.PodCheckpointReady); cond != nil &&
		(cond.Status == metav1.ConditionTrue ||
			cond.Reason == checkpointv1alpha1.PodCheckpointReasonFailed ||
			cond.Reason == checkpointv1alpha1.PodCheckpointReasonSourcePodReplaced) {
		return nil
	}

	// Only act on checkpoints whose source pod this kubelet runs. A
	// PodCheckpoint event can arrive before the pod manager observes an assigned
	// pod, so distinguish that ordering window from a pod that is gone or belongs
	// to another node.
	if pc.Spec.SourcePod == nil || pc.Spec.SourcePod.Name == "" {
		// Validation rejects an unset source pod reference; nothing to act on.
		return nil
	}
	sourcePodName := pc.Spec.SourcePod.Name
	pod, ok := kl.podManager.GetPodByName(namespace, sourcePodName)
	if !ok {
		if kl.kubeClient == nil {
			return nil
		}
		apiPod, err := kl.kubeClient.CoreV1().Pods(namespace).Get(ctx, sourcePodName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return fmt.Errorf("failed to determine node assignment for source pod %q: %w", sourcePodName, err)
		}
		if apiPod.Spec.NodeName == string(kl.nodeName) {
			return errPodCheckpointPodNotObserved
		}
		return nil
	}

	// Pin the source pod instance by UID. A pod name can be reused, so a recorded
	// UID that no longer matches the live pod means the original instance was
	// replaced; fail rather than checkpoint the wrong instance.
	if uid := pc.Spec.SourcePod.UID; uid != nil && *uid != pod.UID {
		return kl.writePodCheckpointStatus(ctx, namespace, name, metav1.ConditionFalse,
			checkpointv1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q has UID %q but spec.sourcePod.uid pins it to %q; the original instance was replaced", sourcePodName, pod.UID, *uid),
			nil, nil)
	}
	if pc.Status.SourcePodUID != nil && *pc.Status.SourcePodUID != pod.UID {
		return kl.writePodCheckpointStatus(ctx, namespace, name, metav1.ConditionFalse,
			checkpointv1alpha1.PodCheckpointReasonSourcePodReplaced,
			fmt.Sprintf("source pod %q UID changed from %q to %q; the original instance was replaced", sourcePodName, *pc.Status.SourcePodUID, pod.UID),
			nil, nil)
	}

	// A checkpoint is already running for this pod; wait for it to finalize.
	if _, inFlight := kl.checkpointsInFlight.Load(pod.UID); inFlight {
		return errPodCheckpointInFlight
	}

	// Record the in-progress status together with the node, the pinned UID, and
	// the sanitized source pod template (the authoritative record a restore is
	// validated against).
	sourcePodUID := pod.UID
	if err := kl.writePodCheckpointStatus(ctx, namespace, name, metav1.ConditionFalse,
		checkpointv1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress",
		checkpointutil.SanitizePodTemplate(pod), &sourcePodUID); err != nil {
		return err
	}

	// CheckpointPod validates preconditions synchronously, creates the
	// kubelet-owned output directory, and runs the CRI call in the background.
	// The checkpoint timeout is enforced by the kubelet via the gRPC call
	// deadline inside CheckpointPod, not via a request field. CheckpointPod
	// clamps this to the configured ceiling (PodCheckpointTimeout); an unset
	// TimeoutSeconds (zero timeout here) falls back to that ceiling, so the
	// operation always has a deadline. Validation bounds a set value to
	// [1, MaxTimeoutSeconds].
	var timeout time.Duration
	if pc.Spec.TimeoutSeconds != nil {
		timeout = time.Duration(*pc.Spec.TimeoutSeconds) * time.Second
	}
	if err := kl.CheckpointPod(ctx, pod.UID, kubecontainer.GetPodFullName(pod), namespace, name, pc.UID, timeout, pc.Spec.CheckpointOptions); err != nil {
		if errors.Is(err, errPodCheckpointInFlight) {
			return err
		}
		// A non-nil error is a synchronous setup/precondition failure (the
		// background checkpoint has not started); record it as failed.
		if statusErr := kl.finalizePodCheckpoint(ctx, namespace, name, false, "", fmt.Sprintf("checkpoint failed: %v", err)); statusErr != nil {
			return fmt.Errorf("checkpoint setup failed (%w) and failed to record terminal status: %w", err, statusErr)
		}
		return nil
	}
	return nil
}

// writePodCheckpointStatus sets the PodCheckpoint "Ready" condition and the
// kubelet-owned status fields (nodeName=self, and, when provided, the pinned
// sourcePodUID and the captured pod template) via the status subresource under
// RetryOnConflict. It is used for the in-progress and SourcePodReplaced
// transitions; the terminal Completed/Failed transition is written by
// finalizePodCheckpoint.
func (kl *Kubelet) writePodCheckpointStatus(ctx context.Context, namespace, name string, status metav1.ConditionStatus, reason, message string, template *v1.PodTemplateSpec, sourcePodUID *types.UID) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		obj, err := kl.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		var pc checkpointv1alpha1.PodCheckpoint
		if err := apiruntime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pc); err != nil {
			return fmt.Errorf("failed to read PodCheckpoint %q: %w", name, err)
		}

		apimeta.SetStatusCondition(&pc.Status.Conditions, metav1.Condition{
			Type:               checkpointv1alpha1.PodCheckpointReady,
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

		statusMap, err := apiruntime.DefaultUnstructuredConverter.ToUnstructured(&pc.Status)
		if err != nil {
			return fmt.Errorf("failed to convert status to unstructured: %w", err)
		}
		if err := unstructured.SetNestedField(obj.Object, statusMap, "status"); err != nil {
			return fmt.Errorf("failed to set status: %w", err)
		}
		_, err = kl.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).UpdateStatus(ctx, obj, metav1.UpdateOptions{})
		return err
	})
}

// checkpointedContainerStatuses builds the convenience status list of the
// containers captured in a checkpoint: running restartable init (sidecar)
// containers followed by regular containers, mirroring the selection in
// checkpointPodContainerIDs. Container names are unique within a pod, so a
// single list covers both.
func checkpointedContainerStatuses(spec *v1.PodSpec) []checkpointv1alpha1.PodCheckpointContainerStatus {
	out := make([]checkpointv1alpha1.PodCheckpointContainerStatus, 0, len(spec.InitContainers)+len(spec.Containers))
	for i := range spec.InitContainers {
		if podutil.IsRestartableInitContainer(&spec.InitContainers[i]) {
			out = append(out, checkpointv1alpha1.PodCheckpointContainerStatus{Name: spec.InitContainers[i].Name, Image: new(spec.InitContainers[i].Image)})
		}
	}
	for _, ctr := range spec.Containers {
		out = append(out, checkpointv1alpha1.PodCheckpointContainerStatus{Name: ctr.Name, Image: new(ctr.Image)})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
