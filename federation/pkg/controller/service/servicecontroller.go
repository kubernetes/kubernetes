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
	"sort"
	"sync"
	"time"

	"reflect"

	"github.com/golang/glog"
	//	kube_api "k8s.io/kubernetes/pkg/api"
	//	kube_errors "k8s.io/kubernetes/pkg/api/errors"
	//	kube_cache "k8s.io/kubernetes/pkg/client/cache"
	//	kube_clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	//	kube_unversioned_core "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/federation/pkg/api"
	"k8s.io/kubernetes/federation/pkg/api/errors"
	"k8s.io/kubernetes/federation/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/runtime"
)

const (
	// TODO: quinton: These constants are all copied from pkg/controller/service.
	//                Refactor into a shared library?
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
	// TODO: quinton: This all copied from pkg/controller/service.
	//                Refactor into a shared library?  Not sure yet whether desirable.
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
	// TODO: quinton: This all copied from pkg/controller/service.
	//                Needs to be adapted for federated services.

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
// (like Kubernetes Services and DNS server records for service discovery) in sync with the registry.
// TODO: quinton: This all copied from pkg/controller/service.
//                Split in to pieces, one to create Kubernetes Services, and the other to create and update
//                DNS records.

func New(cloud cloudprovider.Interface, kubeClient clientset.Interface, clusterName string) *ServiceController {
	// TODO: quinton:
	//                 1. cloudprovider.Interface is used for DNS configuration?
	//                 2. need one kubeClient per cluster, or look up based on type of cluster.
	//                 3. Replace clustername with cluster api object reference.
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&unversioned_core.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "federated-service-controller"})

	return &ServiceController{
		cloud:            cloud,
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		cache:            &serviceCache{serviceMap: make(map[string]*cachedService)},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
		// TODO: quinton: We watch for changes to underlying Kubernetes services, not notes.  Update this accordingly.
		nodeLister: cache.StoreToNodeLister{
			Store: cache.NewStore(cache.MetaNamespaceKeyFunc),
		},
	}
}

// Run starts a background goroutine that watches for changes to Federated services
// and ensures that they have Kubernetes services created, updated or deleted appropriately.
// federationSyncPeriod controls how often we check the federation's services to
// ensure that the correct Kubernetes services (and associated DNS entries) exist.
// This is only necessary to fudge over failed watches.
// clusterSyncPeriod controls how often we check the federation's underlying clusters and
// their Kubernetes services to ensure that matching services created independently of the Federation
// (e.g. directly via the underlying cluster's API) are correctly accounted for.

// It's an error to call Run() more than once for a given ServiceController
// object.
func (s *ServiceController) Run(federationSyncPeriod, clusterSyncPeriod time.Duration) error {
	if err := s.init(); err != nil {
		return err
	}

	// We have to make this check because the ListWatch that we use in
	// WatchServices requires Client functions that aren't in the interface
	// for some reason.
	if _, ok := s.kubeClient.(*clientset.Clientset); !ok { // TODO: quinton: Should we use a generic kubeClient here, or a more specific ubeClient?
		return fmt.Errorf("ServiceController only works with real Client objects, but was passed something else satisfying the clientset.Interface.")
	}

	// Get the currently existing set of Federated services and then all future creates
	// and updates of these services.
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

	clusterLW := cache.NewListWatchFromClient(s.kubeClient.(*clientset.Clientset).CoreClient, "clusters", api.NamespaceAll, fields.Everything())
	cache.NewReflector(clusterLW, &api.Cluster{}, s.clusterLister.Store, 0).Run() //TODO: quinton: Was a nodeLister
	go s.clusterSyncLoop(clusterSyncPeriod)                                       // TODO: quinton: was a nodeSyncLoop
	return nil
}

func (s *ServiceController) init() error {
	if s.cloud == nil {
		return fmt.Errorf("ServiceController should not be run without a cloudprovider.")
	}

	dns, ok := s.cloud.DNS()
	if !ok {
		// TODO: quinton: We should degrade in this case, not fail.
		//                Also, the DNS provider need not be provided by the cloud provider. Factor this out later.
		return fmt.Errorf("the cloud provider does not support external DNS servers.")
	}
	s.dns = dns

	/** TODO: quinton: don't need this - delete
	zones, ok := s.cloud.Zones()
	if !ok {
		return fmt.Errorf("the cloud provider does not support zone enumeration, which is required for creating load balancers.")
	}
	zone, err := zones.GetZone()
	if err != nil {
		return fmt.Errorf("failed to get zone from cloud provider, will not be able to create load balancers: %v", err)
	}
	s.zone = zone
	*/
	return nil
}

// Loop infinitely, processing all service updates provided by the queue.
// TODO: quinton: This needs to be factored out into a re-usable library, as
//                most Federated Controllers are going to require very similar logic.
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
	glog.V(2).Infof("Got new %s delta for service: %+v", delta.Type, deltaService)

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
		// TODO: quinton: Split this off into a separate deletion handler function.  This one is getting way too long!  Sins of the past.
		glog.V(2).Infof("Federated Service %v not found, ensuring underlying cluster Services and global DNS records are deleted", namespacedName)
		s.eventRecorder.Event(service, api.EventTypeNormal, "DeletingClusterServices", "Deleting cluster services")
		// TODO: quinton:
		//               1. Need to split this up, as there are multiple underlying services, and multiple DNS records to be deleted
		//               2. Event records probably need to be more explicit, showing service names, dns records etc.
		//               3. Factor this properly so that pieces of it can be called by kube-ctl, with explicit inputs instead of cache delta record.
		err := s.DNS.EnsureDNSRecordsDeleted(deltaService)
		if err != nil {
			message := "Error deleting global DNS records (will retry): " + err.Error()
			s.eventRecorder.Event(deltaService, api.EventTypeWarning, "DeletingGlobalServiceDNSRecordsFailed", message)
			return err, cachedService.nextRetryDelay()
		}
		s.eventRecorder.Event(deltaService, api.EventTypeNormal, "DeletedGlobalServiceDNSRecords", "Deleted global DNS records for service")

		// TODO: quinton: Delete the underlying cluster services here.

		s.cache.delete(namespacedName.String())

		cachedService.resetRetryDelay()
		return nil, doNotRetry
	}

	// Update the cached service (used above for populating synthetic deletes)
	cachedService.lastState = service

	err, retry := s.createDNSRecordsIfNeeded(namespacedName, service, cachedService.appliedState) // TODO: quinton: rename->ensureDNSRecordsExist()
	if err != nil {
		message := "Error creating global service DNS records"
		if retry {
			message += " (will retry): "
		} else {
			message += " (will not retry): "
		}
		message += err.Error()
		s.eventRecorder.Event(service, api.EventTypeWarning, "CreatingGlobalServiceDNSRecordsFailed", message)

		return err, cachedService.nextRetryDelay()
	}

	//TODO: quinton: add ensureClusterServicesExist() here.

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
func (s *ServiceController) ensureDNSRecordsExist(namespacedName types.NamespacedName, service, appliedState *api.Service) (error, bool) {
	if appliedState != nil && !s.needsUpdate(appliedState, service) { //TODO: quinton: This logic looks dodgy - probably needs rewrite.
		glog.Infof("DNS records don't need update for service %s", namespacedName)
		return nil, notRetryable
	}

	// Note: It is safe to just call EnsureDNSRecords.  But, on some clouds that requires a delete & create,
	// which may involve service interruption.  Also, we would like user-friendly events.

	// Save the state so we can avoid a write if it doesn't change
	previousState := api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer) // TODO: quinton: are the DNS names in here?  I think so.

	if !wantsDNSRecords(service) {
		needDelete := true
		if appliedState != nil {
			if !wantsDNSRecords(appliedState) {
				needDelete = false
			}
		} else {
			// If we don't have any cached memory of the DNS records, we have to ask
			// the cloud provider for what it knows about them.
			// Technically EnsureDNSRecordsDeleted can cope, but we want to post meaningful events
			_, exists, err := s.dns.GetDNSRecords(service)
			if err != nil {
				return fmt.Errorf("Error getting DNS records for service %s: %v", namespacedName, err), retryable
			}
			if !exists {
				needDelete = false
			}
		}

		if needDelete {
			glog.Infof("Deleting existing DNS records for service %s that no longer needs them.", namespacedName)
			s.eventRecorder.Event(service, api.EventTypeNormal, "DeletingDNSRecords", "Deleting DNS records")
			if err := s.dns.EnsureDNSRecordsDeleted(service); err != nil {
				return err, retryable
			}
			s.eventRecorder.Event(service, api.EventTypeNormal, "DeletedDNSRecords", "Deleted DNS records")
		}

		service.Status.LoadBalancer = api.LoadBalancerStatus{} // TODO: quinton: is the DNS name in here?
	} else {
		glog.V(2).Infof("Ensuring DNS records for service %s", namespacedName)

		// TODO: We could do a dry-run here if wanted to avoid the spurious cloud-calls & events when we restart

		// The DNS records don't exist yet, so create them.
		// TODO: quinton: Split this up and handle each record separately.
		//                Beware though, creating distant ones before near ones can lead to race condidtions on name resolution.
		//                (e.g. client local resolution fails as DNS record doesn't exist yet, but then tries remote resolution and that succeeds,
		//                so it gets stuck for a while on a remote service shard.
		//                Solutions: either use transactions (where providers suport them, e.g. Google Cloud DNS) or just make
		//                sure that the creation sequence is zone-local, region-local, sub-continent local etc.  The latter probably
		//                works for all cases, so transactions are probably not necessary.
		//                Actually, on further thought, the race condition exists anyway - neither of the above tow really solve it.
		//                Ho hum, have to think harder.
		s.eventRecorder.Event(service, api.EventTypeNormal, "CreatingDNSRecords", "Creating DNS Records")
		err := s.createLoadDNSRecords(service)
		if err != nil {
			return fmt.Errorf("Failed to create DNS records for service %s: %v", namespacedName, err), retryable
		}
		s.eventRecorder.Event(service, api.EventTypeNormal, "CreatedDNSRecords", "Created DNS records")
	}

	// Write the state if changed
	// TODO: Be careful here ... what if there were other changes to the service?
	// TODO: quinton: Doesn't update and/or patch solve the above?
	if !api.LoadBalancerStatusEqual(previousState, &service.Status.LoadBalancer) { // TODO: quinton: only interested in the DNS record here.
		// Not even sure it needs updating - check.
		if err := s.persistUpdate(service); err != nil {
			return fmt.Errorf("Failed to persist updated status to apiserver, even after retries. Giving up: %v", err), notRetryable
		}
	} else {
		glog.V(2).Infof("Not persisting unchanged LoadBalancerStatus to registry.") // TODO: quinton: check/fix this.
	}

	return nil, notRetryable
}

func (s *ServiceController) persistUpdate(service *api.Service) error { // TODO: quinton: looks like this only really updates the load balancer
	// status, not the service as a whole. Rename/refactor accordingly.
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err = s.kubeClient.Core().Services(service.Namespace).UpdateStatus(service)
		if err == nil {
			return nil
		}
		// If the object no longer exists, we don't want to recreate it. Just bail
		// out so that we can process the delete, which we should soon be receiving
		// if we haven't already.
		// TODO: quinton: Check this logic - it looks racy.
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to service '%s/%s' that no longer exists: %v",
				service.Namespace, service.Name, err)
			return nil
		}
		// TODO: Try to resolve the conflict if the change was unrelated to load
		// balancer status. For now, just rely on the fact that we'll
		// also process the update that caused the resource version to change.
		// TODO: quinton: Check this logic - it looks racy.
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

func (s *ServiceController) createDNSRecords(service *api.Service) error { //TODO: quinton: Add equivalent createClusterServices(...)
	clusterServiceLister, err := s.clusterServiceLister.List()
	if err != nil {
		return err
	}

	// - Only one protocol supported per service
	// - Not all cloud providers support all protocols and the next step is expected to return
	//   an error for unsupported protocols
	status, err := s.dns.EnsureDNSRecords(service, hostsFromServiceList(&clusterServices), service.ObjectMeta.Annotations)
	if err != nil {
		return err
	} else {
		service.Status.LoadBalancer = *status // TODO: quinton: Fix this.
	}

	return nil
}

// ListKeys implements the interface required by DeltaFIFO to list the keys we
// already know about.
// TODO: quinton: All this serviceCache stuff is stolen (erhem cut 'n pasted) from Kubernetes - move to a shared library!
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
	// TODO: quinton: Split the DNS Record stuff from the cluster service stuff
	if !wantsDNSRecords(oldService) && !wantsDNSRecords(newService) {
		return false
	}
	if wantsDNSRecords(oldService) != wantsDNSRecords(newService) {
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "Type", "%v -> %v",
			oldService.Spec.Type, newService.Spec.Type)
		return true
	}
	if !portsEqualForLB(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		// TODO: quinton: I think this still works - check
		return true
	}
	if !loadBalancerIPsAreEqual(oldService, newService) {
		// TODO: quinton: I think this still works - check
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "LoadbalancerIP", "%v -> %v",
			oldService.Spec.LoadBalancerIP, newService.Spec.LoadBalancerIP)
		return true
	}
	if len(oldService.Spec.ExternalIPs) != len(newService.Spec.ExternalIPs) {
		// TODO: quinton: I think this still works - check
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "ExternalIP", "Count: %v -> %v",
			len(oldService.Spec.ExternalIPs), len(newService.Spec.ExternalIPs))
		return true
	}
	for i := range oldService.Spec.ExternalIPs {
		// TODO: quinton: I think this still works - check
		if oldService.Spec.ExternalIPs[i] != newService.Spec.ExternalIPs[i] {
			s.eventRecorder.Eventf(newService, api.EventTypeNormal, "ExternalIP", "Added: %v",
				newService.Spec.ExternalIPs[i])
			return true
		}
	}
	if !reflect.DeepEqual(oldService.Annotations, newService.Annotations) {
		// TODO: quinton: I think this still works - check
		return true
	}
	if oldService.UID != newService.UID {
		// TODO: quinton: I think this still works - check
		s.eventRecorder.Eventf(newService, api.EventTypeNormal, "UID", "%v -> %v",
			oldService.UID, newService.UID)
		return true
	}

	return false
}

func (s *ServiceController) DNSNameFor(service *api.Service) string { // TODO: quinton: Need multiple names, one for each zone, region, continent, global etc.
	return cloudprovider.GetDNSNameFor(service) // TODO: quinton: Should not be cloud provider specific.  Write a generic one compliant with the RFC.
}

func getPortsForLB(service *api.Service) ([]*api.ServicePort, error) {
	// TODO: quinton: Probably applies for DNS SVC records.  Come back to this.
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

func intSlicesEqual(x, y []int) bool { // TODO: quinton - this belongs in the utils library, not here.  Move it.
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

func stringSlicesEqual(x, y []string) bool { // TODO: quinton - this belongs in the utils library, not here.  Move it.
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

func hostsFromServiceList(list *api.ServiceList) []string { // TODO: quinton: This probably needs to return IP's or DNS names, not "hostnames".  Rename and recode.
	result := []string{}
	for ix := range list.Items {
		if list.Items[ix].Spec.Unschedulable { // TODO: quinton: In the case of cluster services this relates to the health of the services, not the nodes.  Fix this.
			continue
		}
		result = append(result, list.Items[ix].Name)
	}
	return result
}

//TODO: quinton: Add a good comment/doc
func getClusterServiceConditionPredicate() cache.ClusterServiceConditionPredicate {
	return func(clusterService kube_api.Service) bool {
		if clusterService.Status.Health != OK { //TODO: quinton: Fix the health check logic here.
			return false
		}
		// If we have no info, don't accept
		if len(clusterService.Status.Conditions) == 0 { // TODO: quinton: Check whether this applies
			return false
		}
		for _, cond := range clusterService.Status.Conditions {
			// We consider the cluster service for inclusion in DNS records only when its ServiceReady condition status
			// is ConditionTrue
			// TODO: quinton: Check up on exactly how service status works, and fix.  Service is deemed healthy when it has at least one healthy backend.
			if cond.Type == api.ServiceReady && cond.Status != api.ConditionTrue {
				glog.V(4).Infof("Ignoring cluster service %v with %v condition status %v", clusterService.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// clusterServiceSyncLoop handles updating the load balancers pointed to by all federated service DNS Records
// whenever the set of cluster services in the federation changes.
// TODOd: quinton: Assumed that cluster service load balancers have host (i.e. DNS) names (and IP's?).  Verify and fix as necessary.
//                Also the logic here bundles all hos/DNS names of all services in the federation together - need to handle service one separately.  Rewrite.
func (s *ServiceController) clusterServiceSyncLoop(period time.Duration) {
	var prevHostNames []string
	var servicesToUpdate []*cachedService
	for range time.Tick(period) {
		nodes, err := s.clusterServiceLister.clusterServiceCondition(getClusterServiceConditionPredicate()).List()
		if err != nil {
			glog.Errorf("Failed to retrieve current set of cluster services from cluster service lister: %v", err)
			continue
		}
		newHostNames := hostsFromClusterServiceList(&clusterServices)
		if stringSlicesEqual(newHostNames, prevHostNames) {
			// The set of host names in the services in the federation hasn't changed, but we can retry
			// updating any services that we failed to update last time around.
			servicesToUpdate = s.updateDNSRecords(servicesToUpdate, newHostNames) // TODO: quinton: this is screwed up - see above.
			continue
		}
		glog.Infof("Detected change in list of current cluster service host names. New  set: %v", newHostNames)

		// Try updating all services, and save the ones that fail to try again next
		// round.
		servicesToUpdate = s.cache.allServices()
		numServices := len(servicesToUpdate)
		servicesToUpdate = s.updateDNSRecords(servicesToUpdate, newHostNames)
		glog.Infof("Successfully updated %d out of %d DNS records to direct traffic to the updated set cluster service host/DNS names/IP's",
			numServices-len(servicesToUpdate), numServices)

		prevHostNames = newHostNames
	}
}

// updateDNSRecords updates all existing federated service DNS Records so that
// they will match the list of host names provided.
// Returns the list of services that couldn't be updated.
// TODO: quinton: This is still screwed up.  Each service has a different set of backends (IP, port).  Fix accordingly.
func (s *ServiceController) updateLoadBalancerHosts(services []*cachedService, hosts []string) (servicesToRetry []*cachedService) {
	for _, service := range services {
		func() {
			service.mu.Lock()
			defer service.mu.Unlock()
			// If the applied state is nil, that means it hasn't yet been successfully dealt
			// with by the DNS Record reconciler. We can trust the DNS Record
			// reconciler to ensure the federated service's DNS records are created to target
			// the correct backend service IP's
			if service.appliedState == nil {
				return
			}
			if err := s.lockedUpdateDNSRecords(service.appliedState, hostNames); err != nil {
				glog.Errorf("External error while updating DNS Records: %v.", err)
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// Updates the DNS records of a service, assuming we hold the mutex
// associated with the service.
// TODO: quinton: Still screwed up in the same way as above.  Fix.
func (s *ServiceController) lockedUpdateDNSRecordHostNames(service *api.Service, hostnames []string) error {
	if !wantsDNSRecords(service) {
		return nil
	}

	// This operation doesn't normally take very long (and happens pretty often), so we only record the final event
	err := s.dns.UpdateDNSRecord(service, hostnames)
	if err == nil {
		s.eventRecorder.Event(service, api.EventTypeNormal, "UpdatedDNSRecords", "Updated DNS Record with new hostnames")
		return nil
	}

	// It's only an actual error if the DNS record still exists.
	if _, exists, err := s.dns.GetDNSRecordsFor(service); err != nil {
		glog.Errorf("External error while checking if DNS Record %q exists: name, %v", cloudprovider.GetDNSName(service), err)
	} else if !exists {
		return nil
	}

	s.eventRecorder.Eventf(service, api.EventTypeWarning, "DNSRecordUpdateFailed", "Error updating DNS record with new hostnames %v: %v", hostnames, err)
	return err
}

func wantsDNSRecord(service *api.Service) bool {
	return service.Spec.Type == api.ServiceTypeLoadBalancer // TODO: quinton: I think this makes sense?
}

func LoadBalancerIPsAreEqual(oldService, newService *api.Service) bool {
	return oldService.Spec.LoadBalancerIP == newService.Spec.LoadBalancerIP
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedService) nextRetryDelay() time.Duration { // TODO: quinton: This belongs in a common library like utils.  Move it.
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
