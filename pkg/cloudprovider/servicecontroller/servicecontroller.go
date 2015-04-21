/*
Copyright 2015 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const (
	workerGoroutines = 10

	clientRetryCount    = 5
	clientRetryInterval = 5 * time.Second

	retryable    = true
	notRetryable = false
)

type cachedService struct {
	service *api.Service
	// Ensures only one goroutine can operate on this service at any given time.
	mu sync.Mutex
}

type serviceCache struct {
	mu         sync.Mutex // protects serviceMap
	serviceMap map[string]*cachedService
}

type ServiceController struct {
	cloud       cloudprovider.Interface
	kubeClient  client.Interface
	clusterName string
	balancer    cloudprovider.TCPLoadBalancer
	zone        cloudprovider.Zone
	cache       *serviceCache
}

// New returns a new service controller to keep cloud provider service resources
// (like external load balancers) in sync with the registry.
func New(cloud cloudprovider.Interface, kubeClient client.Interface, clusterName string) *ServiceController {
	return &ServiceController{
		cloud:       cloud,
		kubeClient:  kubeClient,
		clusterName: clusterName,
		cache:       &serviceCache{serviceMap: make(map[string]*cachedService)},
	}
}

// Run starts a background goroutine that watches for changes to services that
// have (or had) externalLoadBalancers=true and ensures that they have external
// load balancers created and deleted appropriately.
func (s *ServiceController) Run() error {
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
	// No delta compressor is needed for the DeltaFIFO queue because we only ever
	// care about the most recent state.
	serviceQueue := cache.NewDeltaFIFO(cache.MetaNamespaceKeyFunc, nil, s.cache)
	lw := cache.NewListWatchFromClient(s.kubeClient.(*client.Client), "services", api.NamespaceAll, fields.Everything())
	cache.NewReflector(lw, &api.Service{}, serviceQueue, 0).Run()
	for i := 0; i < workerGoroutines; i++ {
		go s.watchServices(serviceQueue)
	}
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
			time.Sleep(5 * time.Second)
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
		namespacedName = types.NamespacedName{service.Namespace, service.Name}
		service = cachedService.service
		delta.Object = cachedService.service
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

	// TODO: Handle added, updated, and sync differently?
	switch delta.Type {
	case cache.Added:
		fallthrough
	case cache.Updated:
		fallthrough
	case cache.Sync:
		err, retry := s.createLoadBalancerIfNeeded(namespacedName, service, cachedService.service)
		if err != nil {
			return err, retry
		}
		// Always update the cache upon success
		cachedService.service = service
		s.cache.set(namespacedName.String(), cachedService)
	case cache.Deleted:
		err := s.ensureLBDeleted(service)
		if err != nil {
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
		if cachedService.Spec.CreateExternalLoadBalancer {
			glog.Infof("Deleting existing load balancer for service %s that needs an updated load balancer.", namespacedName)
			if err := s.ensureLBDeleted(cachedService); err != nil {
				return err, retryable
			}
		}
	} else {
		// If we don't have any cached memory of the load balancer and it already
		// exists, optimistically consider our work done.
		// TODO: If we could read the spec of the existing load balancer, we could
		// determine if an update is necessary.
		exists, err := s.balancer.TCPLoadBalancerExists(s.loadBalancerName(service), s.zone.Region)
		if err != nil {
			return fmt.Errorf("Error getting LB for service %s", namespacedName), retryable
		}
		if exists && len(service.Spec.PublicIPs) == 0 {
			// The load balancer exists, but we apparently don't know about its public
			// IPs, so just delete it and recreate it to get back to a sane state.
			// TODO: Ideally the cloud provider interface would return the IP for us.
			glog.Infof("Deleting old LB for service with no public IPs %s", namespacedName)
			if err := s.ensureLBDeleted(service); err != nil {
				return err, retryable
			}
		} else if exists {
			// TODO: Better handle updates for non-cached services, this is optimistic.
			glog.Infof("LB already exists for service %s", namespacedName)
			return nil, notRetryable
		}
	}

	if !service.Spec.CreateExternalLoadBalancer {
		glog.Infof("Not creating LB for service %s that doesn't want one.", namespacedName)
		return nil, notRetryable
	}

	glog.V(2).Infof("Creating LB for service %s", namespacedName)

	// The load balancer doesn't exist yet, so create it.
	publicIPstring := fmt.Sprint(service.Spec.PublicIPs)
	err := s.createExternalLoadBalancer(service)
	if err != nil {
		return fmt.Errorf("failed to create external load balancer for service %s: %v", namespacedName, err), retryable
	}

	if publicIPstring == fmt.Sprint(service.Spec.PublicIPs) {
		glog.Infof("Not persisting unchanged service to registry.")
		return nil, notRetryable
	}

	// If creating the load balancer succeeded, persist the updated service.
	if err = s.persistUpdate(service); err != nil {
		return fmt.Errorf("Failed to persist updated publicIPs to apiserver, even after retries. Giving up: %v", err), notRetryable
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
		// balancers and public IPs. For now, just rely on the fact that we'll
		// also process the update that caused the resource version to change.
		if errors.IsConflict(err) {
			glog.Infof("Not persisting update to service that has been changed since we received it: %v", err)
			return nil
		}
		glog.Warningf("Failed to persist updated PublicIPs to service %s after creating its external load balancer: %v",
			service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (s *ServiceController) createExternalLoadBalancer(service *api.Service) error {
	ports, err := getTCPPorts(service)
	if err != nil {
		return err
	}
	nodes, err := s.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	name := s.loadBalancerName(service)
	if len(service.Spec.PublicIPs) > 0 {
		for _, publicIP := range service.Spec.PublicIPs {
			// TODO: Make this actually work for multiple IPs by using different
			// names for each. For now, we'll just create the first and break.
			endpoint, err := s.balancer.CreateTCPLoadBalancer(name, s.zone.Region, net.ParseIP(publicIP),
				ports, hostsFromNodeList(nodes), service.Spec.SessionAffinity)
			if err != nil {
				return err
			}
			service.Spec.PublicIPs = []string{endpoint}
			break
		}
	} else {
		endpoint, err := s.balancer.CreateTCPLoadBalancer(name, s.zone.Region, nil,
			ports, hostsFromNodeList(nodes), service.Spec.SessionAffinity)
		if err != nil {
			return err
		}
		service.Spec.PublicIPs = []string{endpoint}
	}
	return nil
}

// Ensures that the load balancer associated with the given service is deleted,
// doing the deletion if necessary. Should always be retried upon failure.
func (s *ServiceController) ensureLBDeleted(service *api.Service) error {
	// This is only needed because not all delete load balancer implementations
	// are currently idempotent to the LB not existing.
	if exists, err := s.balancer.TCPLoadBalancerExists(s.loadBalancerName(service), s.zone.Region); err != nil {
		return err
	} else if !exists {
		return nil
	}

	if err := s.balancer.DeleteTCPLoadBalancer(s.loadBalancerName(service), s.zone.Region); err != nil {
		return err
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
	if !oldService.Spec.CreateExternalLoadBalancer && !newService.Spec.CreateExternalLoadBalancer {
		return false
	}
	if oldService.Spec.CreateExternalLoadBalancer != newService.Spec.CreateExternalLoadBalancer {
		return true
	}
	if !portsEqual(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		return true
	}
	if len(oldService.Spec.PublicIPs) != len(newService.Spec.PublicIPs) {
		return true
	}
	for i := range oldService.Spec.PublicIPs {
		if oldService.Spec.PublicIPs[i] != newService.Spec.PublicIPs[i] {
			return true
		}
	}
	return false
}

func (s *ServiceController) loadBalancerName(service *api.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
}

func getTCPPorts(service *api.Service) ([]int, error) {
	ports := []int{}
	for i := range service.Spec.Ports {
		// TODO: Support UDP. Remove the check from the API validation package once
		// it's supported.
		sp := &service.Spec.Ports[i]
		if sp.Protocol != api.ProtocolTCP {
			return nil, fmt.Errorf("external load balancers for non TCP services are not currently supported.")
		}
		ports = append(ports, sp.Port)
	}
	return ports, nil
}

func portsEqual(x, y *api.Service) bool {
	xPorts, err := getTCPPorts(x)
	if err != nil {
		return false
	}
	yPorts, err := getTCPPorts(y)
	if err != nil {
		return false
	}
	if len(xPorts) != len(yPorts) {
		return false
	}
	sort.Ints(xPorts)
	sort.Ints(yPorts)
	for i := range xPorts {
		if xPorts[i] != yPorts[i] {
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
