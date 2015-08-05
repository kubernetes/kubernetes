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

package servicecontroller

import (
	"fmt"
	"net"
	"sort"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const (
	workerGoroutines = 10

	// How long to wait before retrying the processing of a service change.
	// If this changes, the sleep in hack/jenkins/e2e.sh before downing a cluster
	// should be changed appropriately.
	processingRetryInterval = 5 * time.Second

	clientRetryCount    = 5
	clientRetryInterval = 5 * time.Second

	retryable    = true
	notRetryable = false
)

type cachedService struct {
	// The last-known state of the service
	lastState *api.Service
	// The state as successfully applied to the load balancer
	appliedState *api.Service

	// Ensures only one goroutine can operate on this service at any given time.
	mu sync.Mutex
}

type serviceCache struct {
	mu         sync.Mutex // protects serviceMap
	serviceMap map[string]*cachedService
}

type ServiceController struct {
	cloud            cloudprovider.Interface
	kubeClient       client.Interface
	clusterName      string
	balancer         cloudprovider.TCPLoadBalancer
	zone             cloudprovider.Zone
	cache            *serviceCache
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder
	nodeLister       cache.StoreToNodeLister
}

// New returns a new service controller to keep cloud provider service resources
// (like external load balancers) in sync with the registry.
func New(cloud cloudprovider.Interface, kubeClient client.Interface, clusterName string) *ServiceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(kubeClient.Events(""))
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
// have (or had) externalLoadBalancers=true and ensures that they have external
// load balancers created and deleted appropriately.
// serviceSyncPeriod controls how often we check the cluster's services to
// ensure that the correct external load balancers exist.
// nodeSyncPeriod controls how often we check the cluster's nodes to determine
// if external load balancers need to be updated to point to a new set.
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
	if _, ok := s.kubeClient.(*client.Client); !ok {
		return fmt.Errorf("ServiceController only works with real Client objects, but was passed something else satisfying the client Interface.")
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
	lw := cache.NewListWatchFromClient(s.kubeClient.(*client.Client), "services", api.NamespaceAll, fields.Everything())
	cache.NewReflector(lw, &api.Service{}, serviceQueue, serviceSyncPeriod).Run()
	for i := 0; i < workerGoroutines; i++ {
		go s.watchServices(serviceQueue)
	}

	nodeLW := cache.NewListWatchFromClient(s.kubeClient.(*client.Client), "nodes", api.NamespaceAll, fields.Everything())
	cache.NewReflector(nodeLW, &api.Node{}, s.nodeLister.Store, 0).Run()
	go s.nodeSyncLoop(nodeSyncPeriod)
	return nil
}

func (s *ServiceController) init() error {
	if s.cloud == nil {
		return fmt.Errorf("ServiceController should not be run without a cloudprovider.")
	}

	balancer, ok := s.cloud.TCPLoadBalancer()
	if !ok {
		return fmt.Errorf("the cloud provider does not support external TCP load balancers.")
	}
	s.balancer = balancer

	zones, ok := s.cloud.Zones()
	if !ok {
		return fmt.Errorf("the cloud provider does not support zone enumeration, which is required for creating external load balancers.")
	}
	zone, err := zones.GetZone()
	if err != nil {
		return fmt.Errorf("failed to get zone from cloud provider, will not be able to create external load balancers: %v", err)
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
		err, shouldRetry := s.processDelta(delta)
		if shouldRetry {
			// Add the failed service back to the queue so we'll retry it.
			glog.Errorf("Failed to process service delta. Retrying: %v", err)
			time.Sleep(processingRetryInterval)
			serviceQueue.AddIfNotPresent(deltas)
		} else if err != nil {
			util.HandleError(fmt.Errorf("Failed to process service delta. Not retrying: %v", err))
		}
	}
}

// Returns an error if processing the delta failed, along with a boolean
// indicator of whether the processing should be retried.
func (s *ServiceController) processDelta(delta *cache.Delta) (error, bool) {
	service, ok := delta.Object.(*api.Service)
	var namespacedName types.NamespacedName
	var cachedService *cachedService
	if !ok {
		// If the DeltaFIFO saw a key in our cache that it didn't know about, it
		// can send a deletion with an unknown state. Grab the service from our
		// cache for deleting.
		key, ok := delta.Object.(cache.DeletedFinalStateUnknown)
		if !ok {
			return fmt.Errorf("Delta contained object that wasn't a service or a deleted key: %+v", delta), notRetryable
		}
		cachedService, ok = s.cache.get(key.Key)
		if !ok {
			return fmt.Errorf("Service %s not in cache even though the watcher thought it was. Ignoring the deletion.", key), notRetryable
		}
		service = cachedService.lastState
		delta.Object = cachedService.lastState
		namespacedName = types.NamespacedName{service.Namespace, service.Name}
	} else {
		namespacedName.Namespace = service.Namespace
		namespacedName.Name = service.Name
		cachedService = s.cache.getOrCreate(namespacedName.String())
	}
	glog.V(2).Infof("Got new %s delta for service: %+v", delta.Type, service)

	// Ensure that no other goroutine will interfere with our processing of the
	// service.
	cachedService.mu.Lock()
	defer cachedService.mu.Unlock()

	// Update the cached service (used above for populating synthetic deletes)
	cachedService.lastState = service

	// TODO: Handle added, updated, and sync differently?
	switch delta.Type {
	case cache.Added:
		fallthrough
	case cache.Updated:
		fallthrough
	case cache.Sync:
		err, retry := s.createLoadBalancerIfNeeded(namespacedName, service, cachedService.appliedState)
		if err != nil {
			s.eventRecorder.Event(service, "creating loadbalancer failed", err.Error())
			return err, retry
		}
		// Always update the cache upon success.
		// NOTE: Since we update the cached service if and only if we successully
		// processed it, a cached service being nil implies that it hasn't yet
		// been successfully processed.
		cachedService.appliedState = service
		s.cache.set(namespacedName.String(), cachedService)
	case cache.Deleted:
		err := s.balancer.EnsureTCPLoadBalancerDeleted(s.loadBalancerName(service), s.zone.Region)
		if err != nil {
			s.eventRecorder.Event(service, "deleting loadbalancer failed", err.Error())
			return err, retryable
		}
		s.cache.delete(namespacedName.String())
	default:
		glog.Errorf("Unexpected delta type: %v", delta.Type)
	}
	return nil, notRetryable
}

// Returns whatever error occurred along with a boolean indicator of whether it
// should be retried.
func (s *ServiceController) createLoadBalancerIfNeeded(namespacedName types.NamespacedName, service, cachedService *api.Service) (error, bool) {
	if cachedService != nil && !needsUpdate(cachedService, service) {
		glog.Infof("LB already exists and doesn't need update for service %s", namespacedName)
		return nil, notRetryable
	}
	if cachedService != nil {
		// If the service already exists but needs to be updated, delete it so that
		// we can recreate it cleanly.
		if wantsExternalLoadBalancer(cachedService) {
			glog.Infof("Deleting existing load balancer for service %s that needs an updated load balancer.", namespacedName)
			if err := s.balancer.EnsureTCPLoadBalancerDeleted(s.loadBalancerName(cachedService), s.zone.Region); err != nil {
				return err, retryable
			}
		}
	} else {
		// If we don't have any cached memory of the load balancer, we have to ask
		// the cloud provider for what it knows about it.
		status, exists, err := s.balancer.GetTCPLoadBalancer(s.loadBalancerName(service), s.zone.Region)
		if err != nil {
			return fmt.Errorf("Error getting LB for service %s: %v", namespacedName, err), retryable
		}
		if exists && api.LoadBalancerStatusEqual(status, &service.Status.LoadBalancer) {
			glog.Infof("LB already exists with status %s for previously uncached service %s", status, namespacedName)
			return nil, notRetryable
		} else if exists {
			glog.Infof("Deleting old LB for previously uncached service %s whose endpoint %s doesn't match the service's desired IPs %v",
				namespacedName, status, service.Spec.DeprecatedPublicIPs)
			if err := s.balancer.EnsureTCPLoadBalancerDeleted(s.loadBalancerName(service), s.zone.Region); err != nil {
				return err, retryable
			}
		}
	}

	// Save the state so we can avoid a write if it doesn't change
	previousState := api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)

	if !wantsExternalLoadBalancer(service) {
		glog.Infof("Not creating LB for service %s that doesn't want one.", namespacedName)

		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	} else {
		glog.V(2).Infof("Creating LB for service %s", namespacedName)

		// The load balancer doesn't exist yet, so create it.
		err := s.createExternalLoadBalancer(service)
		if err != nil {
			return fmt.Errorf("failed to create external load balancer for service %s: %v", namespacedName, err), retryable
		}
	}

	// Write the state if changed
	// TODO: Be careful here ... what if there were other changes to the service?
	if !api.LoadBalancerStatusEqual(previousState, &service.Status.LoadBalancer) {
		if err := s.persistUpdate(service); err != nil {
			return fmt.Errorf("Failed to persist updated status to apiserver, even after retries. Giving up: %v", err), notRetryable
		}
	} else {
		glog.Infof("Not persisting unchanged LoadBalancerStatus to registry.")
	}

	return nil, notRetryable
}

func (s *ServiceController) persistUpdate(service *api.Service) error {
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err = s.kubeClient.Services(service.Namespace).Update(service)
		if err == nil {
			return nil
		}
		// If the object no longer exists, we don't want to recreate it. Just bail
		// out so that we can process the delete, which we should soon be receiving
		// if we haven't already.
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to service that no longer exists: %v", err)
			return nil
		}
		// TODO: Try to resolve the conflict if the change was unrelated to load
		// balancer status. For now, just rely on the fact that we'll
		// also process the update that caused the resource version to change.
		if errors.IsConflict(err) {
			glog.Infof("Not persisting update to service that has been changed since we received it: %v", err)
			return nil
		}
		glog.Warningf("Failed to persist updated LoadBalancerStatus to service %s after creating its external load balancer: %v",
			service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (s *ServiceController) createExternalLoadBalancer(service *api.Service) error {
	ports, err := getPortsForLB(service)
	if err != nil {
		return err
	}
	nodes, err := s.nodeLister.List()
	if err != nil {
		return err
	}
	name := s.loadBalancerName(service)
	if len(service.Spec.DeprecatedPublicIPs) > 0 {
		for _, publicIP := range service.Spec.DeprecatedPublicIPs {
			// TODO: Make this actually work for multiple IPs by using different
			// names for each. For now, we'll just create the first and break.
			status, err := s.balancer.CreateTCPLoadBalancer(name, s.zone.Region, net.ParseIP(publicIP),
				ports, hostsFromNodeList(&nodes), service.Spec.SessionAffinity)
			if err != nil {
				return err
			} else {
				service.Status.LoadBalancer = *status
			}
			break
		}
	} else {
		status, err := s.balancer.CreateTCPLoadBalancer(name, s.zone.Region, nil,
			ports, hostsFromNodeList(&nodes), service.Spec.SessionAffinity)
		if err != nil {
			return err
		} else {
			service.Status.LoadBalancer = *status
		}
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

func needsUpdate(oldService *api.Service, newService *api.Service) bool {
	if !wantsExternalLoadBalancer(oldService) && !wantsExternalLoadBalancer(newService) {
		return false
	}
	if wantsExternalLoadBalancer(oldService) != wantsExternalLoadBalancer(newService) {
		return true
	}
	if !portsEqualForLB(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		return true
	}
	if len(oldService.Spec.DeprecatedPublicIPs) != len(newService.Spec.DeprecatedPublicIPs) {
		return true
	}
	for i := range oldService.Spec.DeprecatedPublicIPs {
		if oldService.Spec.DeprecatedPublicIPs[i] != newService.Spec.DeprecatedPublicIPs[i] {
			return true
		}
	}
	return false
}

func (s *ServiceController) loadBalancerName(service *api.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
}

func getPortsForLB(service *api.Service) ([]*api.ServicePort, error) {
	ports := []*api.ServicePort{}
	for i := range service.Spec.Ports {
		// TODO: Support UDP. Remove the check from the API validation package once
		// it's supported.
		sp := &service.Spec.Ports[i]
		if sp.Protocol != api.ProtocolTCP {
			return nil, fmt.Errorf("external load balancers for non TCP services are not currently supported.")
		}
		ports = append(ports, sp)
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
	result := make([]string, len(list.Items))
	for ix := range list.Items {
		result[ix] = list.Items[ix].Name
	}
	return result
}

// nodeSyncLoop handles updating the hosts pointed to by all external load
// balancers whenever the set of nodes in the cluster changes.
func (s *ServiceController) nodeSyncLoop(period time.Duration) {
	var prevHosts []string
	var servicesToUpdate []*cachedService
	// TODO: Eliminate the unneeded now variable once we stop compiling in go1.3.
	// It's needed at the moment because go1.3 requires ranges to be assigned to
	// something to compile, and gofmt1.4 complains about using `_ = range`.
	for now := range time.Tick(period) {
		_ = now
		nodes, err := s.nodeLister.List()
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
		glog.Infof("Successfully updated %d out of %d external load balancers to direct traffic to the updated set of nodes",
			numServices-len(servicesToUpdate), numServices)

		prevHosts = newHosts
	}
}

// updateLoadBalancerHosts updates all existing external load balancers so that
// they will match the list of hosts provided.
// Returns the list of services that couldn't be updated.
func (s *ServiceController) updateLoadBalancerHosts(services []*cachedService, hosts []string) (servicesToRetry []*cachedService) {
	for _, service := range services {
		func() {
			service.mu.Lock()
			defer service.mu.Unlock()
			// If the service is nil, that means it hasn't yet been successfully dealt
			// with by the load balancer reconciler. We can trust the load balancer
			// reconciler to ensure the service's load balancer is created to target
			// the correct nodes.
			if service.appliedState == nil {
				return
			}
			if err := s.lockedUpdateLoadBalancerHosts(service.appliedState, hosts); err != nil {
				glog.Errorf("External error while updating TCP load balancer: %v.", err)
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// Updates the external load balancer of a service, assuming we hold the mutex
// associated with the service.
func (s *ServiceController) lockedUpdateLoadBalancerHosts(service *api.Service, hosts []string) error {
	if !wantsExternalLoadBalancer(service) {
		return nil
	}

	name := cloudprovider.GetLoadBalancerName(service)
	err := s.balancer.UpdateTCPLoadBalancer(name, s.zone.Region, hosts)
	if err == nil {
		return nil
	}

	// It's only an actual error if the load balancer still exists.
	if _, exists, err := s.balancer.GetTCPLoadBalancer(name, s.zone.Region); err != nil {
		glog.Errorf("External error while checking if TCP load balancer %q exists: name, %v")
	} else if !exists {
		return nil
	}
	return err
}

func wantsExternalLoadBalancer(service *api.Service) bool {
	return service.Spec.Type == api.ServiceTypeLoadBalancer
}
