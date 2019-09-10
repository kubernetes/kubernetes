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
	"net/http"
	"net/url"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
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
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	apiregistrationv1apihelper "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1"
	informers "k8s.io/kube-aggregator/pkg/client/informers/externalversions/apiregistration/v1"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/controllers"
)

// ServiceResolver knows how to convert a service reference into an actual location.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string, port int32) (*url.URL, error)
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
	// map from service-namespace -> service-name -> apiservice names
	cache map[string]map[string][]string
	// this lock protects operations on the above cache
	cacheLock sync.RWMutex
}

// NewAvailableConditionController returns a new AvailableConditionController.
func NewAvailableConditionController(
	apiServiceInformer informers.APIServiceInformer,
	serviceInformer v1informers.ServiceInformer,
	endpointsInformer v1informers.EndpointsInformer,
	apiServiceClient apiregistrationclient.APIServicesGetter,
	proxyTransport *http.Transport,
	proxyClientCert []byte,
	proxyClientKey []byte,
	serviceResolver ServiceResolver,
) (*AvailableConditionController, error) {
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

	// if a particular transport was specified, use that otherwise build one
	// construct an http client that will ignore TLS verification (if someone owns the network and messes with your status
	// that's not so bad) and sets a very short timeout.  This is a best effort GET that provides no additional information
	restConfig := &rest.Config{
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
			CertData: proxyClientCert,
			KeyData:  proxyClientKey,
		},
	}
	if proxyTransport != nil && proxyTransport.DialContext != nil {
		restConfig.Dial = proxyTransport.DialContext
	}
	transport, err := rest.TransportFor(restConfig)
	if err != nil {
		return nil, err
	}
	c.discoveryClient = &http.Client{
		Transport: transport,
		// the request should happen quickly.
		Timeout: 5 * time.Second,
	}

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

	return c, nil
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

	availableCondition := apiregistrationv1.APIServiceCondition{
		Type:               apiregistrationv1.Available,
		Status:             apiregistrationv1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}

	// local API services are always considered available
	if apiService.Spec.Service == nil {
		apiregistrationv1apihelper.SetAPIServiceCondition(apiService, apiregistrationv1apihelper.NewLocalAvailableAPIServiceCondition())
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	}

	service, err := c.serviceLister.Services(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
	if apierrors.IsNotFound(err) {
		availableCondition.Status = apiregistrationv1.ConditionFalse
		availableCondition.Reason = "ServiceNotFound"
		availableCondition.Message = fmt.Sprintf("service/%s in %q is not present", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
		apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	} else if err != nil {
		availableCondition.Status = apiregistrationv1.ConditionUnknown
		availableCondition.Reason = "ServiceAccessError"
		availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
		apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
		_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
		return err
	}

	if service.Spec.Type == v1.ServiceTypeClusterIP {
		// if we have a cluster IP service, it must be listening on configured port and we can check that
		servicePort := apiService.Spec.Service.Port
		portName := ""
		foundPort := false
		for _, port := range service.Spec.Ports {
			if port.Port == *servicePort {
				foundPort = true
				portName = port.Name
			}
		}
		if !foundPort {
			availableCondition.Status = apiregistrationv1.ConditionFalse
			availableCondition.Reason = "ServicePortError"
			availableCondition.Message = fmt.Sprintf("service/%s in %q is not listening on port %d", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, *apiService.Spec.Service.Port)
			apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		}

		endpoints, err := c.endpointsLister.Endpoints(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
		if apierrors.IsNotFound(err) {
			availableCondition.Status = apiregistrationv1.ConditionFalse
			availableCondition.Reason = "EndpointsNotFound"
			availableCondition.Message = fmt.Sprintf("cannot find endpoints for service/%s in %q", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace)
			apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		} else if err != nil {
			availableCondition.Status = apiregistrationv1.ConditionUnknown
			availableCondition.Reason = "EndpointsAccessError"
			availableCondition.Message = fmt.Sprintf("service/%s in %q cannot be checked due to: %v", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, err)
			apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		}
		hasActiveEndpoints := false
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) == 0 {
				continue
			}
			for _, endpointPort := range subset.Ports {
				if endpointPort.Name == portName {
					hasActiveEndpoints = true
				}
			}
		}
		if !hasActiveEndpoints {
			availableCondition.Status = apiregistrationv1.ConditionFalse
			availableCondition.Reason = "MissingEndpoints"
			availableCondition.Message = fmt.Sprintf("endpoints for service/%s in %q have no addresses with port name %q", apiService.Spec.Service.Name, apiService.Spec.Service.Namespace, portName)
			apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
			_, err := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			return err
		}
	}
	// actually try to hit the discovery endpoint when it isn't local and when we're routing as a service.
	if apiService.Spec.Service != nil && c.serviceResolver != nil {
		attempts := 5
		results := make(chan error, attempts)
		for i := 0; i < attempts; i++ {
			go func() {
				discoveryURL, err := c.serviceResolver.ResolveEndpoint(apiService.Spec.Service.Namespace, apiService.Spec.Service.Name, *apiService.Spec.Service.Port)
				if err != nil {
					results <- err
					return
				}
				discoveryURL.Path = "/apis/" + apiService.Spec.Group + "/" + apiService.Spec.Version

				errCh := make(chan error)
				go func() {
					// be sure to check a URL that the aggregated API server is required to serve
					newReq, err := http.NewRequest("GET", discoveryURL.String(), nil)
					if err != nil {
						errCh <- err
						return
					}

					// setting the system-masters identity ensures that we will always have access rights
					transport.SetAuthProxyHeaders(newReq, "system:kube-aggregator", []string{"system:masters"}, nil)
					resp, err := c.discoveryClient.Do(newReq)
					if resp != nil {
						resp.Body.Close()
						// we should always been in the 200s or 300s
						if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
							errCh <- fmt.Errorf("bad status from %v: %v", discoveryURL, resp.StatusCode)
							return
						}
					}

					errCh <- err
				}()

				select {
				case err = <-errCh:
					if err != nil {
						results <- fmt.Errorf("failing or missing response from %v: %v", discoveryURL, err)
						return
					}

					// we had trouble with slow dial and DNS responses causing us to wait too long.
					// we added this as insurance
				case <-time.After(6 * time.Second):
					results <- fmt.Errorf("timed out waiting for %v", discoveryURL)
					return
				}

				results <- nil
			}()
		}

		var lastError error
		for i := 0; i < attempts; i++ {
			lastError = <-results
			// if we had at least one success, we are successful overall and we can return now
			if lastError == nil {
				break
			}
		}

		if lastError != nil {
			availableCondition.Status = apiregistrationv1.ConditionFalse
			availableCondition.Reason = "FailedDiscoveryCheck"
			availableCondition.Message = lastError.Error()
			apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
			_, updateErr := updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
			if updateErr != nil {
				return updateErr
			}
			// force a requeue to make it very obvious that this will be retried at some point in the future
			// along with other requeues done via service change, endpoint change, and resync
			return lastError
		}
	}

	availableCondition.Reason = "Passed"
	availableCondition.Message = "all checks passed"
	apiregistrationv1apihelper.SetAPIServiceCondition(apiService, availableCondition)
	_, err = updateAPIServiceStatus(c.apiServiceClient, originalAPIService, apiService)
	return err
}

// updateAPIServiceStatus only issues an update if a change is detected.  We have a tight resync loop to quickly detect dead
// apiservices.  Doing that means we don't want to quickly issue no-op updates.
func updateAPIServiceStatus(client apiregistrationclient.APIServicesGetter, originalAPIService, newAPIService *apiregistrationv1.APIService) (*apiregistrationv1.APIService, error) {
	if equality.Semantic.DeepEqual(originalAPIService.Status, newAPIService.Status) {
		return newAPIService, nil
	}

	newAPIService, err := client.APIServices().UpdateStatus(newAPIService)
	if err != nil {
		return nil, err
	}

	// update metrics
	wasAvailable := apiregistrationv1apihelper.IsAPIServiceConditionTrue(originalAPIService, apiregistrationv1.Available)
	isAvailable := apiregistrationv1apihelper.IsAPIServiceConditionTrue(newAPIService, apiregistrationv1.Available)
	if isAvailable != wasAvailable {
		if isAvailable {
			unavailableGauge.WithLabelValues(newAPIService.Name).Set(0.0)
		} else {
			unavailableGauge.WithLabelValues(newAPIService.Name).Set(1.0)

			reason := "UnknownReason"
			if newCondition := apiregistrationv1apihelper.GetAPIServiceConditionByType(newAPIService, apiregistrationv1.Available); newCondition != nil {
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

func (c *AvailableConditionController) addAPIService(obj interface{}) {
	castObj := obj.(*apiregistrationv1.APIService)
	klog.V(4).Infof("Adding %s", castObj.Name)
	if castObj.Spec.Service != nil {
		c.rebuildAPIServiceCache()
	}
	c.queue.Add(castObj.Name)
}

func (c *AvailableConditionController) updateAPIService(oldObj, newObj interface{}) {
	castObj := newObj.(*apiregistrationv1.APIService)
	oldCastObj := oldObj.(*apiregistrationv1.APIService)
	klog.V(4).Infof("Updating %s", oldCastObj.Name)
	if !reflect.DeepEqual(castObj.Spec.Service, oldCastObj.Spec.Service) {
		c.rebuildAPIServiceCache()
	}
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
	if castObj.Spec.Service != nil {
		c.rebuildAPIServiceCache()
	}
	c.queue.Add(castObj.Name)
}

func (c *AvailableConditionController) getAPIServicesFor(obj runtime.Object) []string {
	metadata, err := meta.Accessor(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}
	c.cacheLock.RLock()
	defer c.cacheLock.RUnlock()
	return c.cache[metadata.GetNamespace()][metadata.GetName()]
}

// if the service/endpoint handler wins the race against the cache rebuilding, it may queue a no-longer-relevant apiservice
// (which will get processed an extra time - this doesn't matter),
// and miss a newly relevant apiservice (which will get queued by the apiservice handler)
func (c *AvailableConditionController) rebuildAPIServiceCache() {
	apiServiceList, _ := c.apiServiceLister.List(labels.Everything())
	newCache := map[string]map[string][]string{}
	for _, apiService := range apiServiceList {
		if apiService.Spec.Service == nil {
			continue
		}
		if newCache[apiService.Spec.Service.Namespace] == nil {
			newCache[apiService.Spec.Service.Namespace] = map[string][]string{}
		}
		newCache[apiService.Spec.Service.Namespace][apiService.Spec.Service.Name] = append(newCache[apiService.Spec.Service.Namespace][apiService.Spec.Service.Name], apiService.Name)
	}

	c.cacheLock.Lock()
	defer c.cacheLock.Unlock()
	c.cache = newCache
}

// TODO, think of a way to avoid checking on every service manipulation

func (c *AvailableConditionController) addService(obj interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Service)) {
		c.queue.Add(apiService)
	}
}

func (c *AvailableConditionController) updateService(obj, _ interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Service)) {
		c.queue.Add(apiService)
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
		c.queue.Add(apiService)
	}
}

func (c *AvailableConditionController) addEndpoints(obj interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Endpoints)) {
		c.queue.Add(apiService)
	}
}

func (c *AvailableConditionController) updateEndpoints(obj, _ interface{}) {
	for _, apiService := range c.getAPIServicesFor(obj.(*v1.Endpoints)) {
		c.queue.Add(apiService)
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
		c.queue.Add(apiService)
	}
}
