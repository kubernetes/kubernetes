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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
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

type ServiceController struct {
	cloud            cloudprovider.Interface
	kubeClient       client.Interface
	clusterName      string
	balancer         cloudprovider.LoadBalancer
	zone             cloudprovider.Zone
	cache            *serviceCache
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder
	// queue for service that need sync
	serviceQueue *workqueue.Type
	// node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister
	// service framework and store
	serviceController *framework.Controller
	serviceStore      cache.StoreToNodeLister
}

// New returns a new service controller to keep cloud provider service resources
// (like load balancers) in sync with the registry.
func New(cloud cloudprovider.Interface, kubeClient client.Interface, clusterName string, serviceSyncPeriod, nodeSyncPeriod time.Duration) *ServiceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(kubeClient.Events(""))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "service-controller"})

	sc := &ServiceController{
		cloud:            cloud,
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		cache:            &serviceCache{serviceMap: make(map[string]*cachedService)},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
	}

	sc.serviceStore.Store, sc.serviceController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return sc.kubeClient.Services(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return sc.kubeClient.Services(api.NamespaceAll).Watch(options)
			},
		},
		&api.Service{},
		serviceSyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				sc.serviceQueue.Add(&cache.Delta{Type: cache.Added, Object: obj})
			},
			UpdateFunc: func(old, cur interface{}) {
				sc.serviceQueue.Add(&cache.Delta{Type: cache.Updated, Object: cur})
			},
			DeleteFunc: func(obj interface{}) {
				sc.serviceQueue.Add(&cache.Delta{Type: cache.Deleted, Object: obj})
			},
		},
	)

	sc.nodeStore.Store, sc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return sc.kubeClient.Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return sc.kubeClient.Nodes().Watch(options)
			},
		},
		&api.Node{},
		nodeSyncPeriod,
		framework.ResourceEventHandlerFuncs{},
	)

	return sc
}

// Run starts a background goroutine that watches for changes to services that
// have (or had) LoadBalancers=true and ensures that they have
// load balancers created and deleted appropriately.
// nodeSyncPeriod controls how often we check the cluster's nodes to determine
// if load balancers need to be updated to point to a new set.
//
// It's an error to call Run() more than once for a given ServiceController
// object.
func (s *ServiceController) Run(nodeSyncPeriod time.Duration) error {
	if err := s.init(); err != nil {
		return err
	}

	// We have to make this check beecause the ListWatch that we use in
	// WatchServices requires Client functions that aren't in the interface
	// for some reason.
	if _, ok := s.kubeClient.(*client.Client); !ok {
		return fmt.Errorf("ServiceController only works with real Client objects, but was passed something else satisfying the client Interface.")
	}

	go s.serviceController.Run(util.NeverStop)
	for i := 0; i < workerGoroutines; i++ {
		go s.serviceWorker()
	}

	go s.nodeController.Run(util.NeverStop)
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
func (s *ServiceController) serviceWorker() {
	for {
		func() {
			item, quit := s.serviceQueue.Get()
			if quit {
				return
			}
			defer s.serviceQueue.Done(item)

			delta, ok := item.(*cache.Delta)
			if delta == nil {
				glog.Errorf("Received nil delta from watcher queue.")
				return
			}
			if !ok {
				glog.Errorf("Received object from service watcher that wasn't Deltas: %+v", item)
				return
			}
			err, shouldRetry := s.processDelta(delta)
			if shouldRetry {
				// Add the failed service back to the queue so we'll retry it.
				glog.Errorf("Failed to process service delta. Retrying: %v", err)
				time.Sleep(processingRetryInterval)
				s.serviceQueue.Add(item)
			} else if err != nil {
				util.HandleError(fmt.Errorf("Failed to process service delta. Not retrying: %v", err))
			}
		}()
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
		namespacedName = types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
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
			return err, retry
		}
		// Always update the cache upon success.
		// NOTE: Since we update the cached service if and only if we successfully
		// processed it, a cached service being nil implies that it hasn't yet
		// been successfully processed.
		cachedService.appliedState = service
		s.cache.set(namespacedName.String(), cachedService)
	case cache.Deleted:
		s.eventRecorder.Event(service, api.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
		err := s.balancer.EnsureLoadBalancerDeleted(s.loadBalancerName(service), s.zone.Region)
		if err != nil {
			message := "Error deleting load balancer (will retry): " + err.Error()
			s.eventRecorder.Event(service, api.EventTypeWarning, "DeletingLoadBalancerFailed", message)
			return err, retryable
		}
		s.eventRecorder.Event(service, api.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
		s.cache.delete(namespacedName.String())
	default:
		glog.Errorf("Unexpected delta type: %v", delta.Type)
	}
	return nil, notRetryable
}

// Returns whatever error occurred along with a boolean indicator of whether it
// should be retried.
func (s *ServiceController) createLoadBalancerIfNeeded(namespacedName types.NamespacedName, service, appliedState *api.Service) (error, bool) {
	if appliedState != nil && !needsUpdate(appliedState, service) {
		glog.Infof("LB already exists and doesn't need update for service %s", namespacedName)
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
		err := s.createLoadBalancer(service)
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
			glog.V(4).Infof("Not persisting update to service that has been changed since we received it: %v", err)
			return nil
		}
		glog.Warningf("Failed to persist updated LoadBalancerStatus to service %s after creating its load balancer: %v",
			service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (s *ServiceController) createLoadBalancer(service *api.Service) error {
	ports, err := getPortsForLB(service)
	if err != nil {
		return err
	}
	nodes, err := s.nodeStore.List()
	if err != nil {
		return err
	}
	name := s.loadBalancerName(service)
	// - Only one protocol supported per service
	// - Not all cloud providers support all protocols and the next step is expected to return
	//   an error for unsupported protocols
	status, err := s.balancer.EnsureLoadBalancer(name, s.zone.Region, net.ParseIP(service.Spec.LoadBalancerIP),
		ports, hostsFromNodeList(&nodes), service.Spec.SessionAffinity)
	if err != nil {
		return err
	} else {
		service.Status.LoadBalancer = *status
	}

	return nil
}

func (s *ServiceController) loadBalancerName(service *api.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
}

// nodeSyncLoop handles updating the hosts pointed to by all load
// balancers whenever the set of nodes in the cluster changes.
func (s *ServiceController) nodeSyncLoop(period time.Duration) {
	var prevHosts []string
	var servicesToUpdate []*cachedService
	for range time.Tick(period) {
		nodes, err := s.nodeStore.NodeCondition(getNodeConditionPredicate()).List()
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
