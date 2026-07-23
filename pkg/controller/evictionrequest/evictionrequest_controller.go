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

package evictionrequest

import (
	"context"
	"sync"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	lifecycleinformers "k8s.io/client-go/informers/lifecycle/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	lifecyclelisters "k8s.io/client-go/listers/lifecycle/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/evictionrequest/metrics"
	"k8s.io/utils/clock"
)

type EvictionRequestController struct {
	controllerName string
	kubeClient     clientset.Interface

	evictionLister       lifecyclelisters.EvictionLister
	evictionListerSynced cache.InformerSynced

	evictionRequestLister       lifecyclelisters.EvictionRequestLister
	evictionRequestListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	// evictionIndexer allows looking up evictions by target UID
	evictionIndexer cache.Indexer

	// evictionRequestIndexer allows looking up eviction requests by target UID
	evictionRequestIndexer cache.Indexer

	// evictionQueue is the work queue for Eviction reconciliation.
	// Handles responder selection.
	evictionQueue workqueue.TypedRateLimitingInterface[string]

	// evictionRequestQueue is the work queue for EvictionRequest reconciliation.
	// Handles Eviction creation.
	evictionRequestQueue workqueue.TypedRateLimitingInterface[string]

	// evictionMetaRefreshQueue generates responder and requester labels, as well as owner references,
	// and adds them to Evictions.
	// 1. This is kept separate because UpdateStatus() blocks metadata updates (see
	//    evictionStatusStrategy.GetResetFields). We use Server-Side Apply in a separate
	//    evictionQueue to work around this.
	// 2. Not every evictionQueue sync requires an evictionMetaRefreshQueue sync.
	evictionMetaRefreshQueue workqueue.TypedRateLimitingInterface[string]

	// syncEvictionHandler is the function called to sync an Eviction.
	// It may be replaced during tests.
	syncEvictionHandler func(ctx context.Context, key string) error

	// syncEvictionRequestHandler is the function called to sync an EvictionRequest.
	// It may be replaced during tests.
	syncEvictionRequestHandler func(ctx context.Context, key string) error

	// syncEvictionMetaRefreshHandler is the function called to refresh an Eviction metadata.
	// It may be replaced during tests.
	syncEvictionMetaRefreshHandler func(ctx context.Context, key string) error

	// clock is used for time-based operations (e.g., checking heartbeat timeouts).
	// It may be replaced during tests with a fake clock.
	clock clock.PassiveClock
}

// NewController creates a new eviction request controller.
func NewController(
	ctx context.Context,
	evictionInformer lifecycleinformers.EvictionInformer,
	evictionRequestInformer lifecycleinformers.EvictionRequestInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
	controllerName string,
) (*EvictionRequestController, error) {

	metrics.Register()

	c := &EvictionRequestController{
		controllerName:              controllerName,
		kubeClient:                  kubeClient,
		evictionLister:              evictionInformer.Lister(),
		evictionListerSynced:        evictionInformer.Informer().HasSynced,
		evictionRequestLister:       evictionRequestInformer.Lister(),
		evictionRequestListerSynced: evictionRequestInformer.Informer().HasSynced,
		podLister:                   podInformer.Lister(),
		podListerSynced:             podInformer.Informer().HasSynced,
		evictionQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction",
			},
		),
		evictionRequestQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction_request",
			},
		),
		evictionMetaRefreshQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction_metadata_refresh",
			},
		),
		clock: clock.RealClock{},
	}

	c.syncEvictionHandler = c.syncEviction
	c.syncEvictionRequestHandler = c.syncEvictionRequest
	c.syncEvictionMetaRefreshHandler = c.syncEvictionMetaRefresh

	return c, nil
}

// Run starts the eviction request controller.
func (c *EvictionRequestController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting controller", "controller", c.controllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", c.controllerName)
		c.evictionQueue.ShutDown()
		c.evictionRequestQueue.ShutDown()
		c.evictionMetaRefreshQueue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podListerSynced, c.evictionListerSynced, c.evictionRequestListerSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runEvictionRequestWorker, time.Second)
		})
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runEvictionWorker, time.Second)
		})
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, c.runEvictionRefreshWorker, time.Second)
	})

	<-ctx.Done()
}

// runEvictionWorker processes items from the evictionQueue.
func (c *EvictionRequestController) runEvictionWorker(ctx context.Context) {
	for c.processNextEvictionWorkItem(ctx) {
	}
}

// runEvictionRefreshWorker processes items from the evictionMetaRefreshQueue.
func (c *EvictionRequestController) runEvictionRefreshWorker(ctx context.Context) {
	for c.processEvictionRefreshWorkItem(ctx) {
	}
}

// runEvictionProcessingWorker processes items from the evictionRequestQueue.
func (c *EvictionRequestController) runEvictionRequestWorker(ctx context.Context) {
	for c.processNextEvictionRequestWorkItem(ctx) {
	}
}

// processNextEvictionWorkItem processes the next item from the evictionQueue.
func (c *EvictionRequestController) processNextEvictionWorkItem(ctx context.Context) bool {
	key, quit := c.evictionQueue.Get()
	if quit {
		return false
	}
	defer c.evictionQueue.Done(key)

	err := c.syncEvictionHandler(ctx, key)
	if err == nil {
		c.evictionQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync Eviction", "key", key)
	c.evictionQueue.AddRateLimited(key)
	return true
}

// processEvictionRefreshWorkItem processes the next item from the evictionMetaRefreshQueue.
func (c *EvictionRequestController) processEvictionRefreshWorkItem(ctx context.Context) bool {
	key, quit := c.evictionMetaRefreshQueue.Get()
	if quit {
		return false
	}
	defer c.evictionMetaRefreshQueue.Done(key)

	err := c.syncEvictionMetaRefreshHandler(ctx, key)
	if err == nil {
		c.evictionMetaRefreshQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync Eviction metadata refresh", "key", key)
	c.evictionMetaRefreshQueue.AddRateLimited(key)
	return true
}

// processEvictionRefreshWorkItem processes the next item from the evictionRequestQueue.
func (c *EvictionRequestController) processNextEvictionRequestWorkItem(ctx context.Context) bool {
	key, quit := c.evictionRequestQueue.Get()
	if quit {
		return false
	}
	defer c.evictionRequestQueue.Done(key)

	err := c.syncEvictionRequestHandler(ctx, key)
	if err == nil {
		c.evictionRequestQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest", "key", key)
	c.evictionRequestQueue.AddRateLimited(key)
	return true
}

func (c *EvictionRequestController) syncEviction(ctx context.Context, key string) error {
	return nil
}

func (c *EvictionRequestController) syncEvictionRequest(ctx context.Context, key string) error {
	return nil
}

func (c *EvictionRequestController) syncEvictionMetaRefresh(ctx context.Context, key string) error {
	return nil
}
