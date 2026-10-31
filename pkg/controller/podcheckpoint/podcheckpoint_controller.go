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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// Controller manages the lifecycle of PodCheckpoint objects.
//
// The controller assigns each checkpoint to the node running its source Pod,
// so kubelets can watch only their own checkpoints. The kubelet executes the
// checkpoint and finalizes its status (see Kubelet.startPodCheckpointWatch).
// Garbage collection of checkpoint objects/archives is a follow-up.
type Controller struct {
	kubeClient clientset.Interface
	recorder   record.EventRecorder
	podLister  corelisters.PodLister
	podsSynced cache.InformerSynced
	queue      workqueue.TypedRateLimitingInterface[string]
	informer   cache.Controller
	store      cache.Store
}

// NewController creates a new PodCheckpoint lifecycle controller. It watches
// PodCheckpoint objects (via the typed client) and reads source Pods from the
// shared informer so it can assign checkpoints.
func NewController(kubeClient clientset.Interface, podInformer coreinformers.PodInformer) *Controller {
	c := &Controller{
		kubeClient: kubeClient,
		podLister:  podInformer.Lister(),
		podsSynced: podInformer.Informer().HasSynced,
		queue:      workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
	}

	listWatcher := &cache.ListWatch{
		ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
			return kubeClient.NodeV1alpha1().PodCheckpoints(metav1.NamespaceAll).List(ctx, options)
		},
		WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
			return kubeClient.NodeV1alpha1().PodCheckpoints(metav1.NamespaceAll).Watch(ctx, options)
		},
	}
	store, controller := cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: listWatcher,
		ObjectType:    &nodev1alpha1.PodCheckpoint{},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc:    c.enqueueCheckpoint,
			UpdateFunc: func(_, newObj interface{}) { c.enqueueCheckpoint(newObj) },
		},
	})
	c.store = store
	c.informer = controller

	return c
}

func (c *Controller) enqueueCheckpoint(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	c.queue.Add(key)
}

// Run starts the controller.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	eventBroadcaster.StartStructuredLogging(3)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})
	c.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "podcheckpoint-controller"})
	defer eventBroadcaster.Shutdown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting PodCheckpoint controller")
	defer logger.Info("Shutting down PodCheckpoint controller")

	go c.informer.Run(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced, c.podsSynced) {
		logger.Error(nil, "Failed to sync PodCheckpoint controller caches")
		return
	}

	for range workers {
		go wait.UntilWithContext(ctx, c.worker, time.Second)
	}

	<-ctx.Done()
}

func (c *Controller) worker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	if err := c.syncHandler(ctx, key); err != nil {
		utilruntime.HandleError(fmt.Errorf("sync %q failed: %w", key, err))
		c.queue.AddRateLimited(key)
		return true
	}
	c.queue.Forget(key)
	return true
}

// syncHandler assigns pending checkpoints. Decisions start from the informer
// caches; API reads happen only before a write that the caches cannot justify
// on their own.
func (c *Controller) syncHandler(ctx context.Context, key string) error {
	obj, exists, err := c.store.GetByKey(key)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	checkpoint, ok := obj.(*nodev1alpha1.PodCheckpoint)
	if !ok {
		return fmt.Errorf("expected PodCheckpoint for key %q, got %T", key, obj)
	}

	// Return stalled API work to the queue so other checkpoints can be routed.
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	return c.assignCheckpoint(ctx, checkpoint)
}

// assignCheckpoint publishes routing before the kubelet can observe the object
// through its node-scoped watch. Keep an established claim even if the source
// Pod has since disappeared or its name has been reused.
//
// The first attempt uses the cached checkpoint. A conflict means the cache was
// stale, so later attempts read the latest object before deciding again.
func (c *Controller) assignCheckpoint(ctx context.Context, cached *nodev1alpha1.PodCheckpoint) error {
	current := cached
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if current == nil {
			latest, err := c.kubeClient.NodeV1alpha1().PodCheckpoints(cached.Namespace).Get(ctx, cached.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return nil
			}
			if err != nil {
				return err
			}
			current = latest
		}

		if current.UID != cached.UID || current.DeletionTimestamp != nil ||
			current.Status.NodeName != nil || current.Status.SourcePodUID != nil {
			return nil
		}
		condition := apimeta.FindStatusCondition(current.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
		if condition != nil && (condition.Status == metav1.ConditionTrue ||
			condition.Reason == nodev1alpha1.PodCheckpointReasonFailed || condition.Reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced) {
			return nil
		}
		ref := current.Spec.SourcePod
		if ref == nil || ref.Name == "" {
			return nil
		}

		pod, err := c.getSourcePod(ctx, current.Namespace, ref)
		if err != nil {
			return fmt.Errorf("get source Pod %s/%s for checkpoint assignment: %w", current.Namespace, ref.Name, err)
		}
		update := current.DeepCopy()
		failureMessage := ""
		if ref.UID != nil && *ref.UID != pod.UID {
			failureMessage = fmt.Sprintf("source Pod %q has UID %q, not requested UID %q", ref.Name, pod.UID, *ref.UID)
			apimeta.SetStatusCondition(&update.Status.Conditions, metav1.Condition{
				Type:               nodev1alpha1.PodCheckpointConditionReady,
				Status:             metav1.ConditionFalse,
				Reason:             nodev1alpha1.PodCheckpointReasonSourcePodReplaced,
				Message:            failureMessage,
				ObservedGeneration: update.Generation,
			})
		} else {
			if pod.Spec.NodeName == "" || pod.UID == "" {
				return fmt.Errorf("source Pod %s/%s must have a UID and be assigned to a node before checkpoint assignment", pod.Namespace, pod.Name)
			}
			if pod.DeletionTimestamp != nil {
				return fmt.Errorf("source Pod %s/%s is being deleted", pod.Namespace, pod.Name)
			}
			update.Status.NodeName = new(pod.Spec.NodeName)
			update.Status.SourcePodUID = new(pod.UID)
		}

		updated, err := c.kubeClient.NodeV1alpha1().PodCheckpoints(update.Namespace).UpdateStatus(ctx, update, metav1.UpdateOptions{})
		if err != nil {
			current = nil
			return err
		}
		if failureMessage != "" {
			c.recorder.Event(&v1.ObjectReference{
				APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Kind: "PodCheckpoint",
				Namespace: updated.Namespace, Name: updated.Name, UID: updated.UID, ResourceVersion: updated.ResourceVersion,
			}, v1.EventTypeWarning, nodev1alpha1.PodCheckpointReasonFailed, failureMessage)
		}
		return nil
	})
}

// getSourcePod returns the cached source Pod when it can be assigned. The cache
// can lag behind a Pod that was just created, scheduled, or replaced, so any
// other outcome is decided from a live read. This keeps a stale cache from
// failing the checkpoint with SourcePodReplaced.
func (c *Controller) getSourcePod(ctx context.Context, namespace string, ref *nodev1alpha1.PodReference) (*v1.Pod, error) {
	pod, err := c.podLister.Pods(namespace).Get(ref.Name)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	if err == nil && (ref.UID == nil || *ref.UID == pod.UID) &&
		pod.UID != "" && pod.Spec.NodeName != "" && pod.DeletionTimestamp == nil {
		return pod, nil
	}
	return c.kubeClient.CoreV1().Pods(namespace).Get(ctx, ref.Name, metav1.GetOptions{})
}
