/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package service

import (
	"fmt"
	"net"
	"sort"
	"sync"
	"time"

	"reflect"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
)

const (
	workerGoroutines = 10

	// How long to wait before retrying the processing of a service change.
	// If this changes, the sleep in hack/jenkins/e2e.sh before downing a cluster
	// should be changed appropriately.
	minRetryDelay = 5 * time.Second
	maxRetryDelay = 300 * time.Second

	clientRetryCount    = 5
	clientRetryInterval = 5 * time.Second

	retryable    = true
	notRetryable = false

	doNotRetry = time.Duration(0)
)

type cachedService struct {
	// The last-known state of the service
	lastState *api.Service
	// The state as successfully applied to the load balancer
	appliedState *api.Service

	// Ensures only one goroutine can operate on this service at any given time.
	mu sync.Mutex

	// Controls error back-off
	lastRetryDelay time.Duration
}

type serviceCache struct {
	mu         sync.Mutex // protects serviceMap
	serviceMap map[string]*cachedService
}

type ServiceController struct {
	cloud            cloudprovider.Interface
	kubeClient       clientset.Interface
	clusterName      string
	balancer         cloudprovider.LoadBalancer
	zone             cloudprovider.Zone
	cache            *serviceCache
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder
	nodeLister       cache.StoreToNodeLister
}

// New returns a new service controller to keep cloud provider service resources
// (like load balancers) in sync with the registry.
func New(cloud cloudprovider.Interface, kubeClient clientset.Interface, clusterName string) *ServiceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{kubeClient.Core().Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "service-controller"})

	return &ServiceController{
		cloud:            cloud,
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		cache:            &serviceCache{serviceMap: make(map[string]*cachedService)},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
		nodeLister: cache.StoreToNodeLister{
			Store: cache.NewStore(cache.MetaNamespaceKeyFunc),
		},
	}
}

// Run starts a background goroutine that watches for changes to services that
// have (or had) LoadBalancers=true and ensures that they have
// load balancers created and deleted appropriately.
// serviceSyncPeriod controls how often we check the cluster's services to
// ensure that the correct load balancers exist.
// nodeSyncPeriod controls how often we check the cluster's nodes to determine
// if load balancers need to be updated to point to a new set.
//
// It's an error to call Run() more than once for a given ServiceController
// object.
func (s *ServiceController) Run(serviceSyncPeriod, nodeSyncPeriod time.Duration) error {
	if err := s.init(); err != nil {
		return err
	}

	// We have to make this check beecause the ListWatch that we use in
	// WatchServices requires Client functions that aren't in the interface
	// for some reason.
	if _, ok := s.kubeClient.(*clientset.Clientset); !ok {
		return fmt.Errorf("ServiceController only works with real Client objects, but was passed something else satisfying the clientset.Interface.")
	}

	// Get the currently existing set of services and then all future creates
	// and updates of services.
	// A delta compressor is needed for the DeltaFIFO queue because we only ever
	// care about the most recent state.
	serviceQueue := cache.NewDeltaFIFO(
		cache.MetaNamespaceKeyFunc,
		cache.DeltaCompressorFunc(func(d cache.Deltas) cache.Deltas {
			if len(d) == 0 {
				return d
			}
			return cache.Deltas{*d.Newest()}
		}),
		s.cache,
	)
	lw := cache.NewListWatchFromClient(s.kubeClient.(*clientset.Clientset).CoreClient, "services", api.NamespaceAll, fields.Everything())
	cache.NewReflector(lw, &api.Service{}, serviceQueue, serviceSyncPeriod).Run()
	for i := 0; i < workerGoroutines; i++ {
		go s.watchServices(serviceQueue)
	}

	nodeLW := cache.NewListWatchFromClient(s.kubeClient.(*clientset.Clientset).CoreClient, "nodes", api.NamespaceAll, fields.Everything())
	cache.NewReflector(nodeLW, &api.Node{}, s.nodeLister.Store, 0).Run()
	go s.nodeSyncLoop(nodeSyncPeriod)
	return nil
}

func (s *ServiceController) init() error {
	if s.cloud == nil {
		return fmt.Errorf("ServiceController should not be run without a cloudprovider.")
	}

	balancer, ok := s.cloud.LoadBalancer()
	if !ok {
		return fmt.Errorf("the cloud provider does not support external load balancers.")
	}
	s.balancer = balancer

	zones, ok := s.cloud.Zones()
	if !ok {
		return fmt.Errorf("the cloud provider does not support zone enumeration, which is required for creating load balancers.")
	}
	zone, err := zones.GetZone()
	if err != nil {
		return fmt.Errorf("failed to get zone from cloud provider, will not be able to create load balancers: %v", err)
	}
	s.zone = zone
	return nil
}

// Loop infinitely, processing all service updates provided by the queue.
func (s *ServiceController) watchServices(serviceQueue *cache.DeltaFIFO) {
	for {
		newItem := serviceQueue.Pop()
		deltas, ok := newItem.(cache.Deltas)
		if !ok {
			glog.Errorf("Received object from service watcher that wasn't Deltas: %+v", newItem)
		}
		delta := deltas.Newest()
		if delta == nil {
			glog.Errorf("Received nil delta from watcher queue.")
			continue
		}
		err, retryDelay := s.processDelta(delta)
		if retryDelay != 0 {
			// Add the failed service back to the queue so we'll retry it.
			glog.Errorf("Failed to process service delta. Retrying in %s: %v", retryDelay, err)
			go func(deltas cache.Deltas, delay time.Duration) {
				time.Sleep(delay)
				if err := serviceQueue.AddIfNotPresent(deltas); err != nil {
					glog.Errorf("Error requeuing service delta - will not retry: %v", err)
				}
			}(deltas, retryDelay)
		} else if err != nil {
			runtime.HandleError(fmt.Errorf("Failed to process service delta. Not retrying: %v", err))
		}
	}
}

// Returns an error if processing the delta failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry in that Duration.
func (s *ServiceController) processDelta(delta *cache.Delta) (error, time.Duration) {
	deltaService, ok := delta.Object.(*api.Service)
	var namespacedName types.NamespacedName
	var cachedService *cachedService
	if !ok {
		// If the DeltaFIFO saw a key in our cache that it didn't know about, it
		// can send a deletion with an unknown state. Grab the service from our
		// cache for deleting.
		key, ok := delta.Object.(cache.DeletedFinalStateUnknown)
		if !ok {
			return fmt.Errorf("Delta contained object that wasn't a service or a deleted key: %+v", delta), doNotRetry
		}
		cachedService, ok = s.cache.get(key.Key)
		if !ok {
			return fmt.Errorf("Service %s not in cache even though the watcher thought it was. Ignoring the deletion.", key), doNotRetry
		}
		deltaService = cachedService.lastState
		delta.Object = deltaService
		namespacedName = types.NamespacedName{Namespace: deltaService.Namespace, Name: deltaService.Name}
	} else {
		namespacedName.Namespace = deltaService.Namespace
		namespacedName.Name = deltaService.Name
		cachedService = s.cache.getOrCreate(namespacedName.String())
	}
	glog.V(2).Infof("Got new %s delta for service: %v", delta.Type, namespacedName)

	// Ensure that no other goroutine will interfere with our processing of the
	// service.
	cachedService.mu.Lock()
	defer cachedService.mu.Unlock()

	// Get the most recent state of the service from the API directly rather than
	// trusting the body of the delta. This avoids update re-ordering problems.
	// TODO: Handle sync delta types differently rather than doing a get on every
	// service every time we sync?
	service, err := s.kubeClient.Core().Services(namespacedName.Namespace).Get(namespacedName.Name)
	if err != nil && !errors.IsNotFound(err) {
		glog.Warningf("Failed to get most recent state of service %v from API (will retry): %v", namespacedName, err)
		return err, cachedService.nextRetryDelay()
	} else if errors.IsNotFound(err) {
		glog.V(2).Infof("Service %v not found, ensuring load balancer is deleted", namespacedName)
		s.eventRecorder.Event(deltaService, api.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
		err := s.balancer.EnsureLoadBalancerDeleted(s.loadBalancerName(deltaService), s.zone.Region)
		if err != nil {
			message := "Error deleting load balancer (will retry): " + err.Error()
			s.eventRecorder.Event(deltaService, api.EventTypeWarning, "DeletingLoadBalancerFailed", message)
			return err, cachedService.nextRetryDelay()
		}
		s.eventRecorder.Event(deltaService, api.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
		s.cache.delete(namespacedName.String())

		cachedService.resetRetryDelay()
		return nil, doNotRetry
	}

	// Update the cached service (used above for populating synthetic deletes)
	cachedService.lastState = service

	err, retry := s.createLoadBalancerIfNeeded(namespacedName, service, cachedService.appliedState)
	if err != nil {
		message := "Error creating load balancer"
		if retry {
			message += " (will retry): "
		} else {
			message += " (will not retry): "
		}
		message += err.Error()
		s.eventRecorder.Event(service, api.EventTypeWarning, "CreatingLoadBalancerFailed", message)

		return err, cachedService.nextRetryDelay()
	}
	// Always update the cache upon success.
	// NOTE: Since we update the cached service if and only if we successfully
	// processed it, a cached service being nil implies that it hasn't yet
	// been successfully processed.
	cachedService.appliedState = service
	s.cache.set(namespacedName.String(), cachedService)

	cachedService.resetRetryDelay()
	return nil, doNotRetry
}

// Returns whatever error occurred along with a boolean indicator of whether it
// should be retried.
func (s *ServiceController) createLoadBalancerIfNeeded(namespacedName types.NamespacedName, service, appliedState *api.Service) (error, bool) {
	if appliedState != nil && !s.needsUpdate(appliedState, service) {
		glog.Infof("LB doesn't need update for service %s", namespacedName)
		return nil, notRetryable
	}

	// Note: It is safe to just call EnsureLoadBalancer.  But, on some clouds that requires a delete & create,
	// which may involve service interruption.  Also, we would like user-friendly events.

	// Save the state so we can avoid a write if it doesn't change
	previousState := api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)

	if !wantsLoadBalancer(service) {
		needDelete := true
		if appliedState != nil {
			if !wantsLoadBalancer(appliedState) {
				needDelete = false
			}
		} else {
			// If we don't have any cached memory of the load balancer, we have to ask
			// the cloud provider for what it knows about it.
			// Technically EnsureLoadBalancerDeleted can cope, but we want to post meaningful events
			_, exists, err := s.balancer.GetLoadBalancer(s.loadBalancerName(service), s.zone.Region)
			if err != nil {
				return fmt.Errorf("Error getting LB for service %s: %v", namespacedName, err), retryable
			}
			if !exists {
				needDelete = false
			}
		}

		if needDelete {
			glog.Infof("Deleting existing load balancer for service %s that no longer needs a load balancer.", namespacedName)
			s.eventRecorder.Event(service, api.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
			if err := s.balancer.EnsureLoadBalancerDeleted(s.loadBalancerName(service), s.zone.Region); err != nil {
				return err, retryable
			}
			s.eventRecorder.Event(service, api.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
		}

		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	} else {
		glog.V(2).Infof("Ensuring LB for service %s", namespacedName)

		// TODO: We could do a dry-run here if wanted to avoid the spurious cloud-calls & events when we restart

		// The load balancer doesn't exist yet, so create it.
		s.eventRecorder.Event(service, api.EventTypeNormal, "CreatingLoadBalancer", "Creating load balancer")

		err := s.createLoadBalancer(service, namespacedName)
		if err != nil {
			return fmt.Errorf("Failed to create load balancer for service %s: %v", namespacedName, err), retryable
		}
		s.eventRecorder.Event(service, api.EventTypeNormal, "CreatedLoadBalancer", "Created load balancer")
	}

	// Write the state if changed
	// TODO: Be careful here ... what if there were other changes to the service?
	if !api.LoadBalancerStatusEqual(previousState, &service.Status.LoadBalancer) {
		if err := s.persistUpdate(service); err != nil {
			return fmt.Errorf("Failed to persist updated status to apiserver, even after retries. Giving up: %v", err), notRetryable
		}
	} else {
		glog.V(2).Infof("Not persisting unchanged LoadBalancerStatus to registry.")
	}

	return nil, notRetryable
}

func (s *ServiceController) persistUpdate(service *api.Service) error {
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err = s.kubeClient.Core().Services(service.Namespace).UpdateStatus(service)
		if err == nil {
			return nil
		}
		// If the object no longer exists, we don't want to recreate it. Just bail
		// out so that we can process the delete, which we should soon be receiving
		// if we haven't already.
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to service '%s/%s' that no longer exists: %v",
				service.Namespace, service.Name, err)
			return nil
		}
		// TODO: Try to resolve the conflict if the change was unrelated to load
		// balancer status. For now, just rely on the fact that we'll
		// also process the update that caused the resource version to change.
		if errors.IsConflict(err) {
			glog.V(4).Infof("Not persisting update to service '%s/%s' that has been changed since we received it: %v",
				service.Namespace, service.Name, err)
			return nil
		}
		glog.Warningf("Failed to persist updated LoadBalancerStatus to service '%s/%s' after creating its load balancer: %v",
			service.Namespace, service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (s *ServiceController) createLoadBalancer(service *api.Service, serviceName types.NamespacedName) error {
	ports, err := getPortsForLB(service)
	if err != nil {
		return err
	}
	nodes, err := s.nodeLister.List()
	if err != nil {
		return err
	}
	name := s.loadBalancerName(service)
	// - Only one protocol supported per service
	// - Not all cloud providers support all protocols and the next step is expected to return
	//   an error for unsupported protocols
	status, err := s.balancer.EnsureLoadBalancer(name, s.zone.Region, net.ParseIP(service.Spec.LoadBalancerIP),
		ports, hostsFromNodeList(&nodes), serviceName, service.Spec.SessionAffinity, service.ObjectMeta.Annotations)
	if err != nil {
		return err
	} else {
		service.Status.LoadBalancer = *status
	}

	return nil
}

// ListKeys implements the interface required by DeltaFIFO to list the keys we
// already know about.
func (s *serviceCache) ListKeys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	keys := make([]string, 0, len(s.serviceMap))
	for k := range s.serviceMap {
		keys = append(keys, k)
	}
	return keys
}

// GetByKey returns the value stored in the serviceMap under the given key
func (s *serviceCache) GetByKey(key string) (interface{}, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.serviceMap[key]; ok {
		return v, true, nil
	}
	return nil, false, nil
}

// ListKeys implements the interface required by DeltaFIFO to list the keys we
// already know about.
func (s *serviceCache) allServices() []*cachedService {
	s.mu.Lock()
	defer s.mu.Unlock()
	services := make([]*cachedService, 0, len(s.serviceMap))
	for _, v := range s.serviceMap {
		services = append(services, v)
	}
	return services
}

func (s *serviceCache) get(serviceName string) (*cachedService, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	service, ok := s.serviceMap[serviceName]
	return service, ok
}

func (s *serviceCache) getOrCreate(serviceName string) *cachedService {
	s.mu.Lock()
	defer s.mu.Unlock()
	service, ok := s.serviceMap[serviceName]
	if !ok {
		service = &cachedService{}
		s.serviceMap[serviceName] = service
	}
	return service
}

func (s *serviceCache) set(serviceName string, service *cachedService) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.serviceMap[serviceName] = service
}

func (s *serviceCache) delete(serviceName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.serviceMap, serviceName)
}

func (s *ServiceController) needsUpdate(oldService *api.Service, newService *api.Service) bool {
	if !wantsLoadBalancer(oldService) && !wantsLoadBalancer(newService) {
		return false
	}
	if wantsLoadBalancer(oldService) != wantsLoadBalancer(newService) {
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "Type", "%v -> %v",
			oldService.Spec.Type, newService.Spec.Type)
		return true
	}
	if !portsEqualForLB(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		return true
	}
	if !loadBalancerIPsAreEqual(oldService, newService) {
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "LoadbalancerIP", "%v -> %v",
			oldService.Spec.LoadBalancerIP, newService.Spec.LoadBalancerIP)
		return true
	}
	if len(oldService.Spec.ExternalIPs) != len(newService.Spec.ExternalIPs) {
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "ExternalIP", "Count: %v -> %v",
			len(oldService.Spec.ExternalIPs), len(newService.Spec.ExternalIPs))
		return true
	}
	for i := range oldService.Spec.ExternalIPs {
		if oldService.Spec.ExternalIPs[i] != newService.Spec.ExternalIPs[i] {
			s.eventRecorder.Eventf(newService, api.EventTypeNormal, "ExternalIP", "Added: %v",
				newService.Spec.ExternalIPs[i])
			return true
		}
	}
	if !reflect.DeepEqual(oldService.Annotations, newService.Annotations) {
		return true
	}
	if oldService.UID != newService.UID {
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "UID", "%v -> %v",
			oldService.UID, newService.UID)
		return true
	}

	return false
}

func (s *ServiceController) loadBalancerName(service *api.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
}

func getPortsForLB(service *api.Service) ([]*api.ServicePort, error) {
	var protocol api.Protocol

	ports := []*api.ServicePort{}
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]
		// The check on protocol was removed here.  The cloud provider itself is now responsible for all protocol validation
		ports = append(ports, sp)
		if protocol == "" {
			protocol = sp.Protocol
		} else if protocol != sp.Protocol && wantsLoadBalancer(service) {
			// TODO:  Convert error messages to use event recorder
			return nil, fmt.Errorf("mixed protocol external load balancers are not supported.")
		}
	}
	return ports, nil
}

func portsEqualForLB(x, y *api.Service) bool {
	xPorts, err := getPortsForLB(x)
	if err != nil {
		return false
	}
	yPorts, err := getPortsForLB(y)
	if err != nil {
		return false
	}
	return portSlicesEqualForLB(xPorts, yPorts)
}

func portSlicesEqualForLB(x, y []*api.ServicePort) bool {
	if len(x) != len(y) {
		return false
	}

	for i := range x {
		if !portEqualForLB(x[i], y[i]) {
			return false
		}
	}
	return true
}

func portEqualForLB(x, y *api.ServicePort) bool {
	// TODO: Should we check name?  (In theory, an LB could expose it)
	if x.Name != y.Name {
		return false
	}

	if x.Protocol != y.Protocol {
		return false
	}

	if x.Port != y.Port {
		return false
	}

	if x.NodePort != y.NodePort {
		return false
	}

	// We don't check TargetPort; that is not relevant for load balancing
	// TODO: Should we blank it out?  Or just check it anyway?

	return true
}

func intSlicesEqual(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	if !sort.IntsAreSorted(x) {
		sort.Ints(x)
	}
	if !sort.IntsAreSorted(y) {
		sort.Ints(y)
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func stringSlicesEqual(x, y []string) bool {
	if len(x) != len(y) {
		return false
	}
	if !sort.StringsAreSorted(x) {
		sort.Strings(x)
	}
	if !sort.StringsAreSorted(y) {
		sort.Strings(y)
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func hostsFromNodeList(list *api.NodeList) []string {
	result := []string{}
	for ix := range list.Items {
		if list.Items[ix].Spec.Unschedulable {
			continue
		}
		result = append(result, list.Items[ix].Name)
	}
	return result
}

func getNodeConditionPredicate() cache.NodeConditionPredicate {
	return func(node api.Node) bool {
		// We add the master to the node list, but its unschedulable.  So we use this to filter
		// the master.
		// TODO: Use a node annotation to indicate the master
		if node.Spec.Unschedulable {
			return false
		}
		// If we have no info, don't accept
		if len(node.Status.Conditions) == 0 {
			return false
		}
		for _, cond := range node.Status.Conditions {
			// We consider the node for load balancing only when its NodeReady condition status
			// is ConditionTrue
			if cond.Type == api.NodeReady && cond.Status != api.ConditionTrue {
				glog.V(4).Infof("Ignoring node %v with %v condition status %v", node.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// nodeSyncLoop handles updating the hosts pointed to by all load
// balancers whenever the set of nodes in the cluster changes.
func (s *ServiceController) nodeSyncLoop(period time.Duration) {
	var prevHosts []string
	var servicesToUpdate []*cachedService
	for range time.Tick(period) {
		nodes, err := s.nodeLister.NodeCondition(getNodeConditionPredicate()).List()
		if err != nil {
			glog.Errorf("Failed to retrieve current set of nodes from node lister: %v", err)
			continue
		}
		newHosts := hostsFromNodeList(&nodes)
		if stringSlicesEqual(newHosts, prevHosts) {
			// The set of nodes in the cluster hasn't changed, but we can retry
			// updating any services that we failed to update last time around.
			servicesToUpdate = s.updateLoadBalancerHosts(servicesToUpdate, newHosts)
			continue
		}
		glog.Infof("Detected change in list of current cluster nodes. New node set: %v", newHosts)

		// Try updating all services, and save the ones that fail to try again next
		// round.
		servicesToUpdate = s.cache.allServices()
		numServices := len(servicesToUpdate)
		servicesToUpdate = s.updateLoadBalancerHosts(servicesToUpdate, newHosts)
		glog.Infof("Successfully updated %d out of %d load balancers to direct traffic to the updated set of nodes",
			numServices-len(servicesToUpdate), numServices)

		prevHosts = newHosts
	}
}

// updateLoadBalancerHosts updates all existing load balancers so that
// they will match the list of hosts provided.
// Returns the list of services that couldn't be updated.
func (s *ServiceController) updateLoadBalancerHosts(services []*cachedService, hosts []string) (servicesToRetry []*cachedService) {
	for _, service := range services {
		func() {
			service.mu.Lock()
			defer service.mu.Unlock()
			// If the applied state is nil, that means it hasn't yet been successfully dealt
			// with by the load balancer reconciler. We can trust the load balancer
			// reconciler to ensure the service's load balancer is created to target
			// the correct nodes.
			if service.appliedState == nil {
				return
			}
			if err := s.lockedUpdateLoadBalancerHosts(service.appliedState, hosts); err != nil {
				glog.Errorf("External error while updating load balancer: %v.", err)
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// Updates the load balancer of a service, assuming we hold the mutex
// associated with the service.
func (s *ServiceController) lockedUpdateLoadBalancerHosts(service *api.Service, hosts []string) error {
	if !wantsLoadBalancer(service) {
		return nil
	}

	// This operation doesn't normally take very long (and happens pretty often), so we only record the final event
	name := cloudprovider.GetLoadBalancerName(service)
	err := s.balancer.UpdateLoadBalancer(name, s.zone.Region, hosts)
	if err == nil {
		s.eventRecorder.Event(service, api.EventTypeNormal, "UpdatedLoadBalancer", "Updated load balancer with new hosts")
		return nil
	}

	// It's only an actual error if the load balancer still exists.
	if _, exists, err := s.balancer.GetLoadBalancer(name, s.zone.Region); err != nil {
		glog.Errorf("External error while checking if load balancer %q exists: name, %v", name, err)
	} else if !exists {
		return nil
	}

	s.eventRecorder.Eventf(service, api.EventTypeWarning, "LoadBalancerUpdateFailed", "Error updating load balancer with new hosts %v: %v", hosts, err)
	return err
}

func wantsLoadBalancer(service *api.Service) bool {
	return service.Spec.Type == api.ServiceTypeLoadBalancer
}

func loadBalancerIPsAreEqual(oldService, newService *api.Service) bool {
	return oldService.Spec.LoadBalancerIP == newService.Spec.LoadBalancerIP
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedService) nextRetryDelay() time.Duration {
	s.lastRetryDelay = s.lastRetryDelay * 2
	if s.lastRetryDelay < minRetryDelay {
		s.lastRetryDelay = minRetryDelay
	}
	if s.lastRetryDelay > maxRetryDelay {
		s.lastRetryDelay = maxRetryDelay
	}
	return s.lastRetryDelay
}

// Resets the retry exponential backoff.  mutex must be held.
func (s *cachedService) resetRetryDelay() {
	s.lastRetryDelay = time.Duration(0)
}
