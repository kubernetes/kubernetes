/*
Copyright 2017 The Kubernetes Authors.

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

package apiserver

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	v1informers "k8s.io/client-go/informers/core/v1"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion"
	informers "k8s.io/kube-aggregator/pkg/client/informers/internalversion/apiregistration/internalversion"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion"
	"k8s.io/kube-aggregator/pkg/controllers"
)

type AvailableConditionController struct {
	apiServiceClient apiregistrationclient.APIServicesGetter

	apiServiceLister listers.APIServiceLister
	apiServiceSynced cache.InformerSynced

	// serviceLister is used to get the IP to create the transport for
	serviceLister  v1listers.ServiceLister
	servicesSynced cache.InformerSynced

	endpointsLister v1listers.EndpointsLister
	endpointsSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface
}

func NewAvailableConditionController(
	apiServiceInformer informers.APIServiceInformer,
	serviceInformer v1informers.ServiceInformer,
	endpointsInformer v1informers.EndpointsInformer,
	apiServiceClient apiregistrationclient.APIServicesGetter,
) *AvailableConditionController {
	c := &AvailableConditionController{
		apiServiceClient: apiServiceClient,
		apiServiceLister: apiServiceInformer.Lister(),
		apiServiceSynced: apiServiceInformer.Informer().HasSynced,
		serviceLister:    serviceInformer.Lister(),
		servicesSynced:   serviceInformer.Informer().HasSynced,
		endpointsLister:  endpointsInformer.Lister(),
		endpointsSynced:  endpointsInformer.Informer().HasSynced,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "AvailableConditionController"),
	}

	apiServiceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addAPIService,
		UpdateFunc: c.updateAPIService,
		DeleteFunc: c.deleteAPIService,
	})

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addService,
		UpdateFunc: c.updateService,
		DeleteFunc: c.deleteService,
	})

	endpointsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addEndpoints,
		UpdateFunc: c.updateEndpoints,
		DeleteFunc: c.deleteEndpoints,
	})

	c.syncFn = c.sync

	return c
}

func (c *AvailableConditionController) sync(key string) error {
	inAPIService, err := c.apiServiceLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	apiService := inAPIService.DeepCopy()

	availableCondition := apiregistration.APIServiceCondition{
		Type:               apiregistration.Available,
		Status:             apiregistration.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}

	// local API services are always considered available
	if apiService.Spec.Service == nil {
		apiregistration.SetAPIServiceCondition(apiService, apiregistration.NewLocalAvailableAPIServiceCondition())
		_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
		return err
	}

	service, err := c.serviceLister.Services(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
	if apierrors.IsNotFound(err) {
		availableCondition.Status = apiregistration.ConditionFalse
		availableCondition.Reason = "ServiceNotFound"
		availableCondition.Message = fmt.Sprintf("service/%s in %q is not present", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
		apiregistration.SetAPIServiceCondition(apiService, availableCondition)
		_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
		return err
	} else if err != nil {
		availableCondition.Status = apiregistration.ConditionUnknown
		availableCondition.Reason = "ServiceAccessError"
		availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
		apiregistration.SetAPIServiceCondition(apiService, availableCondition)
		_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
		return err
	}

	if service.Spec.Type == v1.ServiceTypeClusterIP {
		endpoints, err := c.endpointsLister.Endpoints(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
		if apierrors.IsNotFound(err) {
			availableCondition.Status = apiregistration.ConditionFalse
			availableCondition.Reason = "EndpointsNotFound"
			availableCondition.Message = fmt.Sprintf("cannot find endpoints for service/%s in %q", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
			return err
		} else if err != nil {
			availableCondition.Status = apiregistration.ConditionUnknown
			availableCondition.Reason = "EndpointsAccessError"
			availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
			return err
		}
		hasActiveEndpoints := false
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) > 0 {
				hasActiveEndpoints = true
				break
			}
		}
		if !hasActiveEndpoints {
			availableCondition.Status = apiregistration.ConditionFalse
			availableCondition.Reason = "MissingEndpoints"
			availableCondition.Message = fmt.Sprintf("endpoints for service/%s in %q have no addresses", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := c.apiServiceClient.APIServices().UpdateStatus(apiService)
			return err
		}
	}

	// TODO actually try to hit the discovery endpoint

	availableCondition.Reason = "Passed"
	availableCondition.Message = "all checks passed"
	apiregistration.SetAPIServiceCondition(apiService, availableCondition)
	_, err = c.apiServiceClient.APIServices().UpdateStatus(apiService)
	return err
}

func (c *AvailableConditionController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	glog.Infof("Starting AvailableConditionController")
	defer glog.Infof("Shutting down AvailableConditionController")

	if !controllers.WaitForCacheSync("AvailableConditionController", stopCh, c.apiServiceSynced, c.servicesSynced, c.endpointsSynced) {
		return
	}

	// only start one worker thread since its a slow moving API and the aggregation server adding bits
	// aren't threadsafe
	go wait.Until(c.runWorker, time.Second, stopCh)

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

	err := c.syncFn(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *AvailableConditionController) enqueue(obj *apiregistration.APIService) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %#v: %v", obj, err)
		return
	}

	c.queue.Add(key)
}

func (c *AvailableConditionController) addAPIService(obj interface{}) {
	castObj := obj.(*apiregistration.APIService)
	glog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *AvailableConditionController) updateAPIService(obj, _ interface{}) {
	castObj := obj.(*apiregistration.APIService)
	glog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *AvailableConditionController) deleteAPIService(obj interface{}) {
	castObj, ok := obj.(*apiregistration.APIService)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiregistration.APIService)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting %q", castObj.Name)
	c.enqueue(castObj)
}

// there aren't very many apiservices, just check them all.
func (c *AvailableConditionController) getAPIServicesFor(obj runtime.Object) []*apiregistration.APIService {
	metadata, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}

	var ret []*apiregistration.APIService
	apiServiceList, _ := c.apiServiceLister.List(labels.Everything())
	for _, apiService := range apiServiceList {
		if apiService.Spec.Service == nil {
			continue
		}
		if apiService.Spec.Service.Namespace == metadata.GetNamespace() && apiService.Spec.Service.Name == metadata.GetName() {
			ret = append(ret, apiService)
		}
	}

	return ret
}

// TODO, think of a way to avoid checking on every service manipulation

func (c *AvailableConditionController) addService(obj interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Service)) {
		c.enqueue(apiService)
	}
}

func (c *AvailableConditionController) updateService(obj, _ interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Service)) {
		c.enqueue(apiService)
	}
}

func (c *AvailableConditionController) deleteService(obj interface{}) {
	castObj, ok := obj.(*v1.Service)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*v1.Service)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	for _, apiService := range c.getAPIServicesFor(castObj) {
		c.enqueue(apiService)
	}
}

func (c *AvailableConditionController) addEndpoints(obj interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Endpoints)) {
		c.enqueue(apiService)
	}
}

func (c *AvailableConditionController) updateEndpoints(obj, _ interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Endpoints)) {
		c.enqueue(apiService)
	}
}

func (c *AvailableConditionController) deleteEndpoints(obj interface{}) {
	castObj, ok := obj.(*v1.Endpoints)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*v1.Endpoints)
		if !ok {
			glog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	for _, apiService := range c.getAPIServicesFor(castObj) {
		c.enqueue(apiService)
	}
}
