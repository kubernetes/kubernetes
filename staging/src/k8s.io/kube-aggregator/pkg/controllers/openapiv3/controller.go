/*
Copyright 2021 The Kubernetes Authors.

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

package openapiv3

import (
	"context"
	"fmt"
	"net/http"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/controllers/openapiv3/aggregator"
)

const (
	successfulUpdateDelay      = time.Minute
	successfulUpdateDelayLocal = time.Second
	failedUpdateMaxExpDelay    = time.Hour
)

type syncAction int

const (
	syncRequeue syncAction = iota
	syncRequeueRateLimited
	syncNothing
)

// AggregationController periodically checks the list of group-versions handled by each APIService and updates the discovery page periodically
type AggregationController struct {
	openAPIAggregationManager aggregator.SpecProxier
	queue                     workqueue.TypedRateLimitingInterface[string]

	// To allow injection for testing.
	syncHandler func(key string) (syncAction, error)
}

// NewAggregationController creates new OpenAPI aggregation controller.
func NewAggregationController(openAPIAggregationManager aggregator.SpecProxier) *AggregationController {
	c := &AggregationController{
		openAPIAggregationManager: openAPIAggregationManager,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedItemExponentialFailureRateLimiter[string](successfulUpdateDelay, failedUpdateMaxExpDelay),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "open_api_v3_aggregation_controller"},
		),
	}

	c.syncHandler = c.sync

	// update each service at least once, also those which are not coming from APIServices, namely local services
	for _, name := range openAPIAggregationManager.GetAPIServiceNames() {
		c.queue.AddAfter(name, time.Second)
	}

	return c
}

// Run is a legacy wrapper that starts the controller.
//
//logcheck:context // RunWithContext should be used instead of Run in code which supports contextual logging.
func (c *AggregationController) Run(stopCh <-chan struct{}) {
	c.RunWithContext(wait.ContextForChannel(stopCh))
}

// RunWithContext starts OpenAPI V3 AggregationController and blocks until the context is cancelled.
func (c *AggregationController) RunWithContext(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting OpenAPI V3 AggregationController")
	defer logger.Info("Shutting down OpenAPI V3 AggregationController")

	go wait.UntilWithContext(ctx, c.runWorker, time.Second)

	<-ctx.Done()
}

func (c *AggregationController) runWorker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for c.processNextWorkItem(logger) {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *AggregationController) processNextWorkItem(logger klog.Logger) bool {
	key, quit := c.queue.Get()
	defer c.queue.Done(key)
	if quit {
		return false
	}

	if aggregator.IsLocalAPIService(key) {
		// for local delegation targets that are aggregated once per second, log at
		// higher level to avoid flooding the log
		logger.V(6).Info("OpenAPI AggregationController: Processing item", "key", key)
	} else {
		logger.V(4).Info("OpenAPI AggregationController: Processing item", "key", key)
	}

	action, err := c.syncHandler(key)
	if err == nil {
		c.queue.Forget(key)
	} else {
		utilruntime.HandleError(fmt.Errorf("loading OpenAPI spec for %q failed with: %v", key, err))
	}

	switch action {
	case syncRequeue:
		if aggregator.IsLocalAPIService(key) {
			logger.V(7).Info("OpenAPI AggregationController: action for local item: Requeue", "key", key, "duration", successfulUpdateDelayLocal)
			c.queue.AddAfter(key, successfulUpdateDelayLocal)
		} else {
			logger.V(7).Info("OpenAPI AggregationController: action for item: Requeue", "key", key)
			c.queue.AddAfter(key, successfulUpdateDelay)
		}
	case syncRequeueRateLimited:
		logger.Info("OpenAPI AggregationController: action for item: Rate Limited Requeue", "key", key)
		c.queue.AddRateLimited(key)
	case syncNothing:
		logger.Info("OpenAPI AggregationController: action for item: Nothing (removed from the queue)", "key", key)
	}

	return true
}

func (c *AggregationController) sync(key string) (syncAction, error) {
	if err := c.openAPIAggregationManager.UpdateAPIServiceSpec(key); err != nil {
		if err == aggregator.ErrAPIServiceNotFound {
			return syncNothing, nil
		}
		return syncRequeueRateLimited, err
	}
	return syncRequeue, nil
}

// AddAPIService adds a new API Service to OpenAPI Aggregation.
func (c *AggregationController) AddAPIService(handler http.Handler, apiService *v1.APIService) {
	if apiService.Spec.Service == nil {
		return
	}
	c.openAPIAggregationManager.AddUpdateAPIService(handler, apiService)
	c.queue.AddAfter(apiService.Name, time.Second)
}

// UpdateAPIService updates API Service's info and handler.
func (c *AggregationController) UpdateAPIService(handler http.Handler, apiService *v1.APIService) {
	if apiService.Spec.Service == nil {
		return
	}
	c.openAPIAggregationManager.AddUpdateAPIService(handler, apiService)
	key := apiService.Name
	if c.queue.NumRequeues(key) > 0 {
		// The item has failed before. Remove it from failure queue and
		// update it in a second
		c.queue.Forget(key)
		c.queue.AddAfter(key, time.Second)
	}
}

// RemoveAPIService removes API Service from OpenAPI Aggregation Controller.
func (c *AggregationController) RemoveAPIService(apiServiceName string) {
	c.openAPIAggregationManager.RemoveAPIServiceSpec(apiServiceName)
	// This will only remove it if it was failing before. If it was successful, processNextWorkItem will figure it out
	// and will not add it again to the queue.
	c.queue.Forget(apiServiceName)
}
