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
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
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

// ServiceResolver knows how to convert a service reference into an actual location.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string) (*url.URL, error)
}

// AvailableConditionController handles checking the availability of registered API services.
type AvailableConditionController struct {
	apiServiceClient apiregistrationclient.APIServicesGetter

	apiServiceLister listers.APIServiceLister
	apiServiceSynced cache.InformerSynced

	// serviceLister is used to get the IP to create the transport for
	serviceLister  v1listers.ServiceLister
	servicesSynced cache.InformerSynced

	endpointsLister v1listers.EndpointsLister
	endpointsSynced cache.InformerSynced

	discoveryClient *http.Client
	serviceResolver ServiceResolver

	// To allow injection for testing.
	syncFn func(key string) error

	queue workqueue.RateLimitingInterface
}

// NewAvailableConditionController returns a new AvailableConditionController.
func NewAvailableConditionController(
	apiServiceInformer informers.APIServiceInformer,
	serviceInformer v1informers.ServiceInformer,
	endpointsInformer v1informers.EndpointsInformer,
	apiServiceClient apiregistrationclient.APIServicesGetter,
	proxyTransport *http.Transport,
	serviceResolver ServiceResolver,
) *AvailableConditionController {
	c := &AvailableConditionController{
		apiServiceClient: apiServiceClient,
		apiServiceLister: apiServiceInformer.Lister(),
		apiServiceSynced: apiServiceInformer.Informer().HasSynced,
		serviceLister:    serviceInformer.Lister(),
		servicesSynced:   serviceInformer.Informer().HasSynced,
		endpointsLister:  endpointsInformer.Lister(),
		endpointsSynced:  endpointsInformer.Informer().HasSynced,
		serviceResolver:  serviceResolver,
		queue: workqueue.NewNamedRateLimitingQueue(
			// We want a fairly tight requeue time.  The controller listens to the API, but because it relies on the routability of the
			// service network, it is possible for an external, non-watchable factor to affect availability.  This keeps
			// the maximum disruption time to a minimum, but it does prevent hot loops.
			workqueue.NewItemExponentialFailureRateLimiter(5*time.Millisecond, 30*time.Second),
			"AvailableConditionController"),
	}

	// construct an http client that will ignore TLS verification (if someone owns the network and messes with your status
	// that's not so bad) and sets a very short timeout.
	discoveryClient := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
		// the request should happen quickly.
		Timeout: 5 * time.Second,
	}
	if proxyTransport != nil {
		discoveryClient.Transport = proxyTransport
	}
	c.discoveryClient = discoveryClient

	// resync on this one because it is low cardinality and rechecking the actual discovery
	// allows us to detect health in a more timely fashion when network connectivity to
	// nodes is snipped, but the network still attempts to route there.  See
	// https://github.com/openshift/origin/issues/17159#issuecomment-341798063
	apiServiceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addAPIService,
			UpdateFunc: c.updateAPIService,
			DeleteFunc: c.deleteAPIService,
		},
		30*time.Second)

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
	originalAPIService, err := c.apiServiceLister.Get(key)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	apiService := originalAPIService.DeepCopy()

	availableCondition := apiregistration.APIServiceCondition{
		Type:               apiregistration.Available,
		Status:             apiregistration.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}

	// local API services are always considered available
	if apiService.Spec.Service == nil {
		apiregistration.SetAPIServiceCondition(apiService, apiregistration.NewLocalAvailableAPIServiceCondition())
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	}

	service, err := c.serviceLister.Services(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
	if apierrors.IsNotFound(err) {
		availableCondition.Status = apiregistration.ConditionFalse
		availableCondition.Reason = "ServiceNotFound"
		availableCondition.Message = fmt.Sprintf("service/%s in %q is not present", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
		apiregistration.SetAPIServiceCondition(apiService, availableCondition)
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	} else if err != nil {
		availableCondition.Status = apiregistration.ConditionUnknown
		availableCondition.Reason = "ServiceAccessError"
		availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
		apiregistration.SetAPIServiceCondition(apiService, availableCondition)
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	}

	if service.Spec.Type == v1.ServiceTypeClusterIP {
		// if we have a cluster IP service, it must be listening on 443 and we can check that
		foundPort := false
		for _, port := range service.Spec.Ports {
			if port.Port == 443 {
				foundPort = true
			}
		}
		if !foundPort {
			availableCondition.Status = apiregistration.ConditionFalse
			availableCondition.Reason = "ServicePortError"
			availableCondition.Message = fmt.Sprintf("service/%s in %q is not listening on port 443", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		}

		endpoints, err := c.endpointsLister.Endpoints(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
		if apierrors.IsNotFound(err) {
			availableCondition.Status = apiregistration.ConditionFalse
			availableCondition.Reason = "EndpointsNotFound"
			availableCondition.Message = fmt.Sprintf("cannot find endpoints for service/%s in %q", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		} else if err != nil {
			availableCondition.Status = apiregistration.ConditionUnknown
			availableCondition.Reason = "EndpointsAccessError"
			availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
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
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		}
	}
	// actually try to hit the discovery endpoint when it isn't local and when we're routing as a service.
	if apiService.Spec.Service != nil && c.serviceResolver != nil {
		discoveryURL, err := c.serviceResolver.ResolveEndpoint(apiService.Spec.Service.Namespace, apiService.Spec.Service.Name)
		if err != nil {
			return err
		}

		errCh := make(chan error)
		go func() {
			resp, err := c.discoveryClient.Get(discoveryURL.String())
			if resp != nil {
				resp.Body.Close()
			}
			errCh <- err
		}()

		select {
		case err = <-errCh:

		// we had trouble with slow dial and DNS responses causing us to wait too long.
		// we added this as insurance
		case <-time.After(6 * time.Second):
			err = fmt.Errorf("timed out waiting for %v", discoveryURL)
		}

		if err != nil {
			availableCondition.Status = apiregistration.ConditionFalse
			availableCondition.Reason = "FailedDiscoveryCheck"
			availableCondition.Message = fmt.Sprintf("no response from %v: %v", discoveryURL, err)
			apiregistration.SetAPIServiceCondition(apiService, availableCondition)
			_, updateErr := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			if updateErr != nil {
				return updateErr
			}
			// force a requeue to make it very obvious that this will be retried at some point in the future
			// along with other requeues done via service change, endpoint change, and resync
			return err
		}
	}

	availableCondition.Reason = "Passed"
	availableCondition.Message = "all checks passed"
	apiregistration.SetAPIServiceCondition(apiService, availableCondition)
	_, err = updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
	return err
}

// updateAPIServiceStatus only issues an update if a change is detected.  We have a tight resync loop to quickly detect dead
// apiservices.  Doing that means we don't want to quickly issue no-op updates.
func updateAPIServiceStatus(client apiregistrationclient.APIServicesGetter, originalAPIService, newAPIService *apiregistration.APIService) (*apiregistration.APIService, error) {
	if equality.Semantic.DeepEqual(originalAPIService.Status, newAPIService.Status) {
		return newAPIService, nil
	}

	newAPIService, err := client.APIServices().UpdateStatus(newAPIService)
	if err != nil {
		return nil, err
	}

	// update metrics
	wasAvailable := apiregistration.IsAPIServiceConditionTrue(originalAPIService, apiregistration.Available)
	isAvailable := apiregistration.IsAPIServiceConditionTrue(newAPIService, apiregistration.Available)
	if isAvailable != wasAvailable {
		if isAvailable {
			unavailableGauge.WithLabelValues(newAPIService.Name).Set(0.0)
		} else {
			unavailableGauge.WithLabelValues(newAPIService.Name).Set(1.0)

			reason := "UnknownReason"
			if newCondition := apiregistration.GetAPIServiceConditionByType(newAPIService, apiregistration.Available); newCondition != nil {
				reason = newCondition.Reason
			}
			unavailableCounter.WithLabelValues(newAPIService.Name, reason).Inc()
		}
	}

	return newAPIService, nil
}

// Run starts the AvailableConditionController loop which manages the availability condition of API services.
func (c *AvailableConditionController) Run(threadiness int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting AvailableConditionController")
	defer klog.Infof("Shutting down AvailableConditionController")

	if !controllers.WaitForCacheSync("AvailableConditionController", stopCh, c.apiServiceSynced, c.servicesSynced, c.endpointsSynced) {
		return
	}

	for i := 0; i < threadiness; i++ {
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
		klog.Errorf("Couldn't get key for object %#v: %v", obj, err)
		return
	}

	c.queue.Add(key)
}

func (c *AvailableConditionController) addAPIService(obj interface{}) {
	castObj := obj.(*apiregistration.APIService)
	klog.V(4).Infof("Adding %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *AvailableConditionController) updateAPIService(obj, _ interface{}) {
	castObj := obj.(*apiregistration.APIService)
	klog.V(4).Infof("Updating %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *AvailableConditionController) deleteAPIService(obj interface{}) {
	castObj, ok := obj.(*apiregistration.APIService)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*apiregistration.APIService)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	klog.V(4).Infof("Deleting %q", castObj.Name)
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
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*v1.Service)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
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
			klog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		castObj, ok = tombstone.Obj.(*v1.Endpoints)
		if !ok {
			klog.Errorf("Tombstone contained object that is not expected %#v", obj)
			return
		}
	}
	for _, apiService := range c.getAPIServicesFor(castObj) {
		c.enqueue(apiService)
	}
}
