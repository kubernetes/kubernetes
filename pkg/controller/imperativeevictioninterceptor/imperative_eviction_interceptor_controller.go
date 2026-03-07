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

package imperativeevictioninterceptor

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	coordinationlisters "k8s.io/client-go/listers/coordination/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

type ImperativeEvictionInterceptorController struct {
	controllerName string

	kubeClient clientset.Interface

	evictionRequestLister       coordinationlisters.EvictionRequestLister
	evictionRequestListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	queue workqueue.TypedRateLimitingInterface[string]

	// syncHandler is the function called to sync an EvictionRequest.
	// It may be replaced during tests.
	syncHandler func(ctx context.Context, key string) error

	// clock is used for time-based operations.
	// It may be replaced during tests with a fake clock.
	clock clock.PassiveClock
}

// NewController creates a new imperative eviction interceptor controller.
func NewController(
	ctx context.Context,
	controllerName string,
	evictionRequestInformer coordinationinformers.EvictionRequestInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
) (*ImperativeEvictionInterceptorController, error) {
	logger := klog.FromContext(ctx)

	c := &ImperativeEvictionInterceptorController{
		controllerName:              controllerName,
		kubeClient:                  kubeClient,
		evictionRequestLister:       evictionRequestInformer.Lister(),
		evictionRequestListerSynced: evictionRequestInformer.Informer().HasSynced,
		podLister:                   podInformer.Lister(),
		podListerSynced:             podInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName,
			},
		),
		clock: clock.RealClock{},
	}

	c.syncHandler = c.sync

	if _, err := evictionRequestInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			c.enqueue(logger, obj)
		},
		UpdateFunc: func(old, new any) { c.enqueue(logger, new) },
		DeleteFunc: func(obj any) { c.enqueue(logger, obj) },
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	if _, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, new any) {
			oldPod := old.(*v1.Pod)
			newPod := new.(*v1.Pod)
			if oldPod.DeletionTimestamp == nil && newPod.DeletionTimestamp != nil {
				c.queue.Add(evictionRequestKey(newPod))
			}
		},
		DeleteFunc: func(obj any) {
			c.deletePod(logger, obj)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	return c, nil
}

func (c *ImperativeEvictionInterceptorController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", c.controllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down", "controller", c.controllerName)
		c.queue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podListerSynced, c.evictionRequestListerSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runWorker, time.Second)
		})
	}

	<-ctx.Done()
}

// enqueue adds an EvictionRequest to the queue.
func (c *ImperativeEvictionInterceptorController) enqueue(logger klog.Logger, obj any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.queue.Add(key)
}

// deletePod enqueues the EvictionRequest for a deleted Pod.
func (c *ImperativeEvictionInterceptorController) deletePod(logger klog.Logger, obj any) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Couldn't get object from tombstone", "object", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Tombstone contained object that is not a Pod", "object", obj)
			return
		}
	}
	c.queue.Add(evictionRequestKey(pod))
}

func (c *ImperativeEvictionInterceptorController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *ImperativeEvictionInterceptorController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest", "key", key)
	c.queue.AddRateLimited(key)
	return true
}

func (c *ImperativeEvictionInterceptorController) sync(ctx context.Context, key string) error {
	return nil
}

func evictionRequestKey(pod metav1.ObjectMetaAccessor) string {
	return pod.GetObjectMeta().GetNamespace() + "/" + string(pod.GetObjectMeta().GetUID())
}
