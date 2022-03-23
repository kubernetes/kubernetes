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
	queue                     workqueue.RateLimitingInterface

	// To allow injection for testing.
	syncHandler func(key string) (syncAction, error)
}

// NewAggregationController creates new OpenAPI aggregation controller.
func NewAggregationController(openAPIAggregationManager aggregator.SpecProxier) *AggregationController {
	c := &AggregationController{
		openAPIAggregationManager: openAPIAggregationManager,
		queue: workqueue.NewNamedRateLimitingQueue(
			workqueue.NewItemExponentialFailureRateLimiter(successfulUpdateDelay, failedUpdateMaxExpDelay),
			"open_api_v3_aggregation_controller",
		),
	}

	c.syncHandler = c.sync

	// update each service at least once, also those which are not coming from APIServices, namely local services
	for _, name := range openAPIAggregationManager.GetAPIServiceNames() {
		c.queue.AddAfter(name, time.Second)
	}

	return c
}

// Run starts OpenAPI AggregationController
func (c *AggregationController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Info("Starting OpenAPI V3 AggregationController")
	defer klog.Info("Shutting down OpenAPI V3 AggregationController")

	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *AggregationController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *AggregationController) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	defer c.queue.Done(key)
	if quit {
		return false
	}

	if aggregator.IsLocalAPIService(key.(string)) {
		// for local delegation targets that are aggregated once per second, log at
		// higher level to avoid flooding the log
		klog.V(6).Infof("OpenAPI AggregationController: Processing item %s", key)
	} else {
		klog.V(4).Infof("OpenAPI AggregationController: Processing item %s", key)
	}

	action, err := c.syncHandler(key.(string))
	if err == nil {
		c.queue.Forget(key)
	} else {
		utilruntime.HandleError(fmt.Errorf("loading OpenAPI spec for %q failed with: %v", key, err))
	}

	switch action {
	case syncRequeue:
		if aggregator.IsLocalAPIService(key.(string)) {
			klog.V(7).Infof("OpenAPI AggregationController: action for local item %s: Requeue after %s.", key, successfulUpdateDelayLocal)
			c.queue.AddAfter(key, successfulUpdateDelayLocal)
		} else {
			klog.V(7).Infof("OpenAPI AggregationController: action for item %s: Requeue.", key)
			c.queue.AddAfter(key, successfulUpdateDelay)
		}
	case syncRequeueRateLimited:
		klog.Infof("OpenAPI AggregationController: action for item %s: Rate Limited Requeue.", key)
		c.queue.AddRateLimited(key)
	case syncNothing:
		klog.Infof("OpenAPI AggregationController: action for item %s: Nothing (removed from the queue).", key)
	}

	return true
}

func (c *AggregationController) sync(key string) (syncAction, error) {
	err := c.openAPIAggregationManager.UpdateAPIServiceSpec(key)
	switch {
	case err != nil:
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
