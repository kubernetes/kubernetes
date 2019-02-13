/*
Copyright 2016 The Kubernetes Authors.

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

package openapi

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/go-openapi/spec"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator"
	"k8s.io/kube-aggregator/pkg/controllers/openapi/download"
)

const (
	successfulUpdateDelay          = time.Minute
	successfulLocalSpecUpdateDelay = time.Second
	failedUpdateMaxExpDelay        = time.Hour

	localDelegateChainNamePrefix  = "k8s_internal_local_delegation_chain_"
	localDelegateChainNamePattern = localDelegateChainNamePrefix + "%010d.%s"
)

type syncAction int

const (
	syncRequeue syncAction = iota
	syncRequeueRateLimited
	syncNothing
)

// AggregationController reacts on changes of APIServices, downloads OpenAPI specs via handlers and merges them.
type AggregationController struct {
	aggregator aggregator.SpecAggregator
	queue      workqueue.RateLimitingInterface
	downloader *download.Downloader

	lock     sync.Mutex
	handlers map[string]http.Handler

	// To allow injection for testing.
	syncHandler func(key string) (syncAction, error)
}

// NewAggregationControllerWithLocalHandlers creates a new OpenAPI aggregation controller with local delegate fake
// handlers in addition which retrieve local OpenAPI specs via the delegate chain. The delegates do not map 1:1 to
// APIServices and we don't even know which local APIService belong to which delegate.
// TODO: add knowledge about which APIService belong to which delegate, and then use the normal APIService update logic
func NewAggregationControllerWithLocalHandlers(downloader *download.Downloader, a aggregator.SpecAggregator, aggregatorSpec *spec.Swagger, delegationTarget server.DelegationTarget) *AggregationControllerWithLocalHandlers {
	// register aggregator spec
	i := 0
	name, service := localFakeService(i)
	utilruntime.Must(a.AddUpdateService(name, service))
	utilruntime.Must(a.UpdateSpec(name, aggregatorSpec, ""))
	i++

	c := &AggregationControllerWithLocalHandlers{
		AggregationController: NewAggregationController(downloader, a),
		delegationTarget:      delegationTarget,
	}

	// register all other specs and handlers
	for delegate := delegationTarget; delegate != nil; delegate = delegate.NextDelegate() {
		handler := delegate.UnprotectedHandler()
		if handler == nil {
			continue
		}
		name, service := localFakeService(i)
		a.AddUpdateService(name, service)
		c.AggregationController.handlers[name] = handler
		// the spec update will come through periodic queuing through the Run method

		i++
	}

	return c
}

// AggregationControllerWithLocalHandlers is ann OpenAPI aggregation controller which updates local specs from
// the delegate chain once a second.
type AggregationControllerWithLocalHandlers struct {
	*AggregationController

	delegationTarget server.DelegationTarget
}

// localFakeServiceName return a name to be used for the i'th local delegate handler.
func localFakeServiceName(i int) string {
	return fmt.Sprintf(localDelegateChainNamePattern, i, "v1")
}

// localFakeService creates a fake APIService to be used for the i'th local delegate handler.
// TODO: add knowledge about which APIService belongs to which delegate and register each delegate's with the correct APIServices
func localFakeService(i int) (string, *apiregistration.APIService) {
	name := localFakeServiceName(i)
	return name, &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       apiregistration.APIServiceSpec{Group: name, Version: "v1"},
	}
}

func (c *AggregationControllerWithLocalHandlers) Run(stopCh <-chan struct{}) {
	// queue local specs once
	i := 1 // skip aggregator
	for delegate := c.delegationTarget; delegate != nil; delegate = delegate.NextDelegate() {
		name, _ := localFakeService(i)
		c.queue.AddAfter(name, time.Second)
		i++
	}

	c.Run(stopCh)
}

// NewAggregationController creates new OpenAPI aggregation controller.
func NewAggregationController(downloader *download.Downloader, openAPIAggregationManager aggregator.SpecAggregator) *AggregationController {
	c := &AggregationController{
		aggregator: openAPIAggregationManager,
		queue: workqueue.NewNamedRateLimitingQueue(
			workqueue.NewItemExponentialFailureRateLimiter(successfulUpdateDelay, failedUpdateMaxExpDelay), "APIServiceOpenAPIAggregationControllerQueue1"),
		downloader: downloader,
		handlers:   map[string]http.Handler{},
	}

	c.syncHandler = c.sync

	return c
}

// Run starts OpenAPI AggregationController
func (c *AggregationController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting OpenAPI AggregationController")
	defer klog.Infof("Shutting down OpenAPI AggregationController")

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

	klog.Infof("OpenAPI AggregationController: Processing item %s", key)

	action, err := c.syncHandler(key.(string))
	if err == nil {
		c.queue.Forget(key)
	} else {
		utilruntime.HandleError(fmt.Errorf("loading OpenAPI spec for %q failed with: %v", key, err))
	}

	switch action {
	case syncRequeue:
		if strings.HasPrefix(key.(string), localDelegateChainNamePrefix) {
			// local specs are checked much more often.
			klog.V(6).Infof("OpenAPI AggregationController: action for local item %s: Requeue in %s.", key, successfulLocalSpecUpdateDelay)
			c.queue.AddAfter(key, successfulLocalSpecUpdateDelay)
		} else {
			klog.Infof("OpenAPI AggregationController: action for remote item %s: Requeue in %s.", key, successfulUpdateDelay)
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
	_, etag, exists := c.aggregator.Spec(key)
	if !exists {
		return syncNothing, nil
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	handler := c.handlers[key]
	if handler == nil {
		return syncNothing, nil
	}

	returnSpec, newEtag, httpStatus, err := c.downloader.Download(handler, etag)
	switch {
	case err != nil:
		return syncRequeueRateLimited, err
	case httpStatus == http.StatusNotModified:
	case httpStatus == http.StatusNotFound || returnSpec == nil:
		return syncRequeueRateLimited, fmt.Errorf("OpenAPI spec does not exist")
	case httpStatus == http.StatusOK:
		if err := c.aggregator.UpdateSpec(key, returnSpec, newEtag); err != nil {
			return syncRequeueRateLimited, err
		}
	}
	return syncRequeue, nil
}

// AddAPIService adds a new API Service to OpenAPI Aggregation.
func (c *AggregationController) AddAPIService(handler http.Handler, apiService *apiregistration.APIService) {
	if apiService.Spec.Service == nil {
		// ignore local services
		return
	}

	c.lock.Lock()
	defer c.lock.Unlock()

	// TODO: combine APIServices from the same aggregated apiserver by choosing the same key
	key := apiService.Name
	if err := c.aggregator.AddUpdateService(key, apiService); err != nil {
		utilruntime.HandleError(fmt.Errorf("adding %q to AggregationController failed with: %v", apiService.Name, err))
	}
	c.handlers[key] = handler

	c.queue.AddAfter(apiService.Name, time.Second)
}

// UpdateAPIService updates API Service's info and handler.
func (c *AggregationController) UpdateAPIService(handler http.Handler, apiService *apiregistration.APIService) {
	if apiService.Spec.Service == nil {
		// ignore local services
		return
	}

	c.lock.Lock()
	defer c.lock.Unlock()

	// TODO: combine APIServices from the same aggregated apiserver by choosing the same name
	key := apiService.Name
	if err := c.aggregator.AddUpdateService(key, apiService); err != nil {
		utilruntime.HandleError(fmt.Errorf("updating %q to AggregationController failed with: %v", apiService.Name, err))
	}
	c.handlers[key] = handler

	if c.queue.NumRequeues(key) > 0 {
		// The item has failed before. Remove it from failure queue and
		// update it in a second
		c.queue.Forget(key)
		c.queue.AddAfter(key, time.Second)
	}
	// Else: The item has been succeeded before and it will be updated soon (after successfulUpdateDelay)
	// we don't add it again as it will cause a duplication of items.
}

// RemoveAPIService removes API Service from OpenAPI Aggregation Controller.
func (c *AggregationController) RemoveAPIService(apiServiceName string) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// split APIService name which always has the version.group pattern
	ns := strings.SplitN(apiServiceName, ".", 2)
	version, group := ns[0], ns[1]

	key := apiServiceName
	if err := c.aggregator.RemoveService(key, schema.GroupVersion{group, version}); err != nil {
		utilruntime.HandleError(fmt.Errorf("removing %q from AggregationController failed with: %v", apiServiceName, err))
	}
	delete(c.handlers, apiServiceName)

	// This will only remove it if it was failing before. If it was successful, processNextWorkItem will figure it out
	// and will not add it again to the queue.
	c.queue.Forget(apiServiceName)
}
