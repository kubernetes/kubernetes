/*
Copyright 2024 The Kubernetes Authors.

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

package external

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	apiregistrationv1apihelper "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1"
	informers "k8s.io/kube-aggregator/pkg/client/informers/externalversions/apiregistration/v1"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/controllers"
	availabilitymetrics "k8s.io/kube-aggregator/pkg/controllers/status/metrics"
)

// AvailableConditionController handles checking the availability of registered local API services.
type AvailableConditionController struct {
	apiServiceClient apiregistrationclient.APIServicesGetter

	apiServiceLister listers.APIServiceLister
	apiServiceSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.TypedRateLimitingInterface[string]

	// metrics registered into legacy registry
	metrics *availabilitymetrics.Metrics
}

// New returns a new local availability AvailableConditionController.
func New(
	apiServiceInformer informers.APIServiceInformer,
	apiServiceClient apiregistrationclient.APIServicesGetter,
	metrics *availabilitymetrics.Metrics,
) (*AvailableConditionController, error) {
	c := &AvailableConditionController{
		apiServiceClient: apiServiceClient,
		apiServiceLister: apiServiceInformer.Lister(),
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			// We want a fairly tight requeue time.  The controller listens to the API, but because it relies on the routability of the
			// service network, it is possible for an external, non-watchable factor to affect availability.  This keeps
			// the maximum disruption time to a minimum, but it does prevent hot loops.
			workqueue.NewTypedItemExponentialFailureRateLimiter[string](5*time.Millisecond, 30*time.Second),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "LocalAvailabilityController"},
		),
		metrics: metrics,
	}

	// resync on this one because it is low cardinality and rechecking the actual discovery
	// allows us to detect health in a more timely fashion when network connectivity to
	// nodes is snipped, but the network still attempts to route there.  See
	// https://github.com/openshift/origin/issues/17159#issuecomment-341798063
	apiServiceHandler, _ := apiServiceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addAPIService,
			UpdateFunc: c.updateAPIService,
			DeleteFunc: c.deleteAPIService,
		},
		30*time.Second)
	c.apiServiceSynced = apiServiceHandler.HasSynced

	c.syncFn = c.sync

	return c, nil
}

func (c *AvailableConditionController) sync(key string) error {
	originalAPIService, err := c.apiServiceLister.Get(key)
	if apierrors.IsNotFound(err) {
		c.metrics.ForgetAPIService(key)
		return nil
	}
	if err != nil {
		return err
	}

	if originalAPIService.Spec.Service != nil {
		// this controller only handles local APIServices
		return nil
	}

	// local API services are always considered available
	apiService := originalAPIService.DeepCopy()
	apiregistrationv1apihelper.SetAPIServiceCondition(apiService, apiregistrationv1apihelper.NewLocalAvailableAPIServiceCondition())
	_, err = c.updateAPIServiceStatus(originalAPIService, apiService)
	return err
}

// updateAPIServiceStatus only issues an update if a change is detected.  We have a tight resync loop to quickly detect dead
// apiservices. Doing that means we don't want to quickly issue no-op updates.
func (c *AvailableConditionController) updateAPIServiceStatus(originalAPIService, newAPIService *apiregistrationv1.APIService) (*apiregistrationv1.APIService, error) {
	// update this metric on every sync operation to reflect the actual state
	c.metrics.SetUnavailableGauge(newAPIService)

	if equality.Semantic.DeepEqual(originalAPIService.Status, newAPIService.Status) {
		return newAPIService, nil
	}

	orig := apiregistrationv1apihelper.GetAPIServiceConditionByType(originalAPIService, apiregistrationv1.Available)
	now := apiregistrationv1apihelper.GetAPIServiceConditionByType(newAPIService, apiregistrationv1.Available)
	unknown := apiregistrationv1.APIServiceCondition{
		Type:   apiregistrationv1.Available,
		Status: apiregistrationv1.ConditionUnknown,
	}
	if orig == nil {
		orig = &unknown
	}
	if now == nil {
		now = &unknown
	}
	if *orig != *now {
		klog.V(2).InfoS("changing APIService availability", "name", newAPIService.Name, "oldStatus", orig.Status, "newStatus", now.Status, "message", now.Message, "reason", now.Reason)
	}

	newAPIService, err := c.apiServiceClient.APIServices().UpdateStatus(context.TODO(), newAPIService, metav1.UpdateOptions{})
	if err != nil {
		return nil, err
	}

	c.metrics.SetUnavailableCounter(originalAPIService, newAPIService)
	return newAPIService, nil
}

// Run starts the AvailableConditionController loop which manages the availability condition of API services.
func (c *AvailableConditionController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Info("Starting LocalAvailability controller")
	defer klog.Info("Shutting down LocalAvailability controller")

	// This waits not just for the informers to sync, but for our handlers
	// to be called; since the handlers are three different ways of
	// enqueueing the same thing, waiting for this permits the queue to
	// maximally de-duplicate the entries.
	if !controllers.WaitForCacheSync("LocalAvailability", stopCh, c.apiServiceSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *AvailableConditionController) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *AvailableConditionController) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %w", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *AvailableConditionController) addAPIService(obj interface{}) {
	castObj := obj.(*apiregistrationv1.APIService)
	klog.V(4).Infof("Adding %s", castObj.Name)
	c.queue.Add(castObj.Name)
}

func (c *AvailableConditionController) updateAPIService(oldObj, _ interface{}) {
	oldCastObj := oldObj.(*apiregistrationv1.APIService)
	klog.V(4).Infof("Updating %s", oldCastObj.Name)
	c.queue.Add(oldCastObj.Name)
}

func (c *AvailableConditionController) deleteAPIService(obj interface{}) {
	castObj, ok := obj.(*apiregistrationv1.APIService)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiregistrationv1.APIService)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	klog.V(4).Infof("Deleting %q", castObj.Name)
	c.queue.Add(castObj.Name)
}
