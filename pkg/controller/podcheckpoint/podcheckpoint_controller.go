/*
Copyright 2026 The Kubernetes Authors.

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
	"slices"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	coreinformers "k8s.io/client-go/informers/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// RestoreLockFinalizer protects a PodCheckpoint from deletion while a Pod is
// actively restoring from it. The pod-snapshot-controller adds it while a
// restore is in progress and removes it once none is, so a delete request is
// honored only after in-flight restores finish (KEP-5823).
const RestoreLockFinalizer = "checkpoint.k8s.io/restore-lock"

var podCheckpointGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// Controller manages the lifecycle of PodCheckpoint objects.
//
// It is deliberately NOT on the checkpoint execution path (KEP-5823): the
// kubelet executes checkpoints by watching PodCheckpoint objects directly and
// finalizes their status (see Kubelet.startPodCheckpointWatch). The controller
// manages PodCheckpoint lifecycle — currently the restore-lock finalizer that
// prevents deleting a checkpoint while a Pod is restoring from it. Garbage
// collection of checkpoint objects/archives is a follow-up.
type Controller struct {
	dynamicClient dynamic.Interface
	podLister     corelisters.PodLister
	podsSynced    cache.InformerSynced
	queue         workqueue.TypedRateLimitingInterface[string]
	informer      cache.Controller
	store         cache.Store
}

// NewController creates a new PodCheckpoint lifecycle controller. It watches
// PodCheckpoint objects (via the dynamic client) and Pods (via the shared
// informer) so it can maintain the restore-lock finalizer.
func NewController(dynamicClient dynamic.Interface, podInformer coreinformers.PodInformer) *Controller {
	c := &Controller{
		dynamicClient: dynamicClient,
		podLister:     podInformer.Lister(),
		podsSynced:    podInformer.Informer().HasSynced,
		queue:         workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
	}

	listWatcher := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return dynamicClient.Resource(podCheckpointGVR).Namespace(metav1.NamespaceAll).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return dynamicClient.Resource(podCheckpointGVR).Namespace(metav1.NamespaceAll).Watch(context.TODO(), options)
		},
	}
	store, controller := cache.NewInformer(
		listWatcher,
		&unstructured.Unstructured{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.enqueueCheckpoint,
			UpdateFunc: func(_, newObj interface{}) { c.enqueueCheckpoint(newObj) },
		},
	)
	c.store = store
	c.informer = controller

	// A Pod referencing a PodCheckpoint via spec.restoreFrom changing state may
	// flip whether a restore is active, so re-reconcile the referenced checkpoint
	// on Pod events.
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.enqueuePod,
		UpdateFunc: func(_, newObj interface{}) { c.enqueuePod(newObj) },
		DeleteFunc: c.enqueuePod,
	})

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

func (c *Controller) enqueuePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			return
		}
	}
	if pod.Spec.RestoreFrom == nil || *pod.Spec.RestoreFrom == "" {
		return
	}
	c.queue.Add(pod.Namespace + "/" + *pod.Spec.RestoreFrom)
}

// Run starts the controller.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.InfoS("Starting PodCheckpoint controller")
	defer klog.InfoS("Shutting down PodCheckpoint controller")

	go c.informer.Run(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced, c.podsSynced) {
		klog.ErrorS(nil, "Failed to sync PodCheckpoint controller caches")
		return
	}

	for i := 0; i < workers; i++ {
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

// syncHandler reconciles the restore-lock finalizer on a PodCheckpoint: it is
// present exactly while a Pod is actively restoring from the checkpoint.
func (c *Controller) syncHandler(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	obj, err := c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	active, err := c.hasActiveRestore(namespace, name)
	if err != nil {
		return err
	}

	finalizers, _, err := unstructured.NestedStringSlice(obj.Object, "metadata", "finalizers")
	if err != nil {
		return fmt.Errorf("failed to read finalizers of PodCheckpoint %q: %w", name, err)
	}
	has := slices.Contains(finalizers, RestoreLockFinalizer)

	switch {
	case active && !has:
		return c.setRestoreLock(ctx, namespace, name, true)
	case !active && has:
		return c.setRestoreLock(ctx, namespace, name, false)
	}
	return nil
}

// hasActiveRestore reports whether any Pod in the namespace is currently
// restoring from the named checkpoint.
func (c *Controller) hasActiveRestore(namespace, name string) (bool, error) {
	pods, err := c.podLister.Pods(namespace).List(labels.Everything())
	if err != nil {
		return false, err
	}
	for _, pod := range pods {
		if restoreActive(pod, name) {
			return true, nil
		}
	}
	return false, nil
}

// restoreActive reports whether the Pod is actively restoring from the named
// checkpoint. A Pod restoring from a checkpoint stays Pending until its sandbox
// and containers are restored and it goes Running; while Pending it still
// depends on the checkpoint data, so the checkpoint must not be deleted. Once
// the Pod is Running (restore complete) or terminal, the dependency is gone.
func restoreActive(pod *v1.Pod, checkpointName string) bool {
	return pod.Spec.RestoreFrom != nil &&
		*pod.Spec.RestoreFrom == checkpointName &&
		pod.Status.Phase == v1.PodPending
}

// setRestoreLock adds or removes the restore-lock finalizer on the PodCheckpoint
// under RetryOnConflict.
func (c *Controller) setRestoreLock(ctx context.Context, namespace, name string, present bool) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		obj, err := c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		finalizers, _, err := unstructured.NestedStringSlice(obj.Object, "metadata", "finalizers")
		if err != nil {
			return fmt.Errorf("failed to read finalizers of PodCheckpoint %q: %w", name, err)
		}
		had := slices.Contains(finalizers, RestoreLockFinalizer)
		switch {
		case present && !had:
			finalizers = append(finalizers, RestoreLockFinalizer)
		case !present && had:
			finalizers = slices.DeleteFunc(finalizers, func(s string) bool { return s == RestoreLockFinalizer })
		default:
			return nil
		}
		if err := unstructured.SetNestedStringSlice(obj.Object, finalizers, "metadata", "finalizers"); err != nil {
			return fmt.Errorf("failed to set finalizers on PodCheckpoint %q: %w", name, err)
		}
		_, err = c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		return err
	})
}
