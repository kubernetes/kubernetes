/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"fmt"
	"sync"
	"time"

	"reflect"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/metrics"
)

const (
	// Interval of synchronizing service status from apiserver
	serviceSyncPeriod = 30 * time.Second
	// Interval of synchronizing node status from apiserver
	nodeSyncPeriod = 100 * time.Second

	// How long to wait before retrying the processing of a service change.
	// If this changes, the sleep in hack/jenkins/e2e.sh before downing a cluster
	// should be changed appropriately.
	minRetryDelay = 5 * time.Second
	maxRetryDelay = 300 * time.Second

	clientRetryCount    = 5
	clientRetryInterval = 5 * time.Second

	// LabelNodeRoleMaster specifies that a node is a master
	// It's copied over to kubeadm until it's merged in core: https://github.com/kubernetes/kubernetes/pull/39112
	LabelNodeRoleMaster = "node-role.kubernetes.io/master"

	// LabelNodeRoleExcludeBalancer specifies that the node should be
	// exclude from load balancers created by a cloud provider.
	LabelNodeRoleExcludeBalancer = "alpha.service-controller.kubernetes.io/exclude-balancer"
)

type cachedService struct {
	// The cached state of the service
	state *v1.Service
}

type serviceCache struct {
	mu         sync.Mutex // protects serviceMap
	serviceMap map[string]*cachedService
}

// ServiceController keeps cloud provider service resources
// (like load balancers) in sync with the registry.
type ServiceController struct {
	cloud               cloudprovider.Interface
	knownHosts          []*v1.Node
	servicesToUpdate    []*v1.Service
	kubeClient          clientset.Interface
	clusterName         string
	balancer            cloudprovider.LoadBalancer
	cache               *serviceCache
	serviceLister       corelisters.ServiceLister
	serviceListerSynced cache.InformerSynced
	eventBroadcaster    record.EventBroadcaster
	eventRecorder       record.EventRecorder
	nodeLister          corelisters.NodeLister
	nodeListerSynced    cache.InformerSynced
	// services that need to be synced
	queue workqueue.RateLimitingInterface
}

// New returns a new service controller to keep cloud provider service resources
// (like load balancers) in sync with the registry.
func New(
	cloud cloudprovider.Interface,
	kubeClient clientset.Interface,
	serviceInformer coreinformers.ServiceInformer,
	nodeInformer coreinformers.NodeInformer,
	clusterName string,
) (*ServiceController, error) {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(glog.Infof)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "service-controller"})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("service_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	s := &ServiceController{
		cloud:            cloud,
		knownHosts:       []*v1.Node{},
		kubeClient:       kubeClient,
		clusterName:      clusterName,
		cache:            &serviceCache{serviceMap: make(map[string]*cachedService)},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
		nodeLister:       nodeInformer.Lister(),
		nodeListerSynced: nodeInformer.Informer().HasSynced,
		queue:            workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(minRetryDelay, maxRetryDelay), "service"),
	}

	serviceInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: s.enqueueService,
			UpdateFunc: func(old, cur interface{}) {
				oldSvc, ok1 := old.(*v1.Service)
				curSvc, ok2 := cur.(*v1.Service)
				if ok1 && ok2 && s.needsUpdate(oldSvc, curSvc) {
					s.enqueueService(cur)
				}
			},
			DeleteFunc: s.enqueueService,
		},
		serviceSyncPeriod,
	)
	s.serviceLister = serviceInformer.Lister()
	s.serviceListerSynced = serviceInformer.Informer().HasSynced

	if err := s.init(); err != nil {
		return nil, err
	}
	return s, nil
}

// obj could be an *v1.Service, or a DeletionFinalStateUnknown marker item.
func (s *ServiceController) enqueueService(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %#v: %v", obj, err)
		return
	}
	s.queue.Add(key)
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
func (s *ServiceController) Run(stopCh <-chan struct{}, workers int) {
	defer runtime.HandleCrash()
	defer s.queue.ShutDown()

	glog.Info("Starting service controller")
	defer glog.Info("Shutting down service controller")

	if !controller.WaitForCacheSync("service", stopCh, s.serviceListerSynced, s.nodeListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(s.worker, time.Second, stopCh)
	}

	go wait.Until(s.nodeSyncLoop, nodeSyncPeriod, stopCh)

	<-stopCh
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (s *ServiceController) worker() {
	for s.processNextWorkItem() {
	}
}

func (s *ServiceController) processNextWorkItem() bool {
	key, quit := s.queue.Get()
	if quit {
		return false
	}
	defer s.queue.Done(key)

	err := s.syncService(key.(string))
	if err == nil {
		s.queue.Forget(key)
		return true
	}

	runtime.HandleError(fmt.Errorf("error processing service %v (will retry): %v", key, err))
	s.queue.AddRateLimited(key)
	return true
}

func (s *ServiceController) init() error {
	if s.cloud == nil {
		return fmt.Errorf("WARNING: no cloud provider provided, services of type LoadBalancer will fail")
	}

	balancer, ok := s.cloud.LoadBalancer()
	if !ok {
		return fmt.Errorf("the cloud provider does not support external load balancers")
	}
	s.balancer = balancer

	return nil
}

// processServiceUpdate operates loadbalancers for the incoming service accordingly.
// Returns an error if processing the service update failed.
func (s *ServiceController) processServiceUpdate(cachedService *cachedService, service *v1.Service, key string) error {
	if cachedService.state != nil {
		if cachedService.state.UID != service.UID {
			err := s.processLoadBalancerDelete(cachedService, key)
			if err != nil {
				return err
			}
		}
	}
	// cache the service, we need the info for service deletion
	cachedService.state = service
	err := s.createLoadBalancerIfNeeded(key, service)
	if err != nil {
		eventType := "CreatingLoadBalancerFailed"
		message := "Error creating load balancer (will retry): "
		if !wantsLoadBalancer(service) {
			eventType = "CleanupLoadBalancerFailed"
			message = "Error cleaning up load balancer (will retry): "
		}
		message += err.Error()
		s.eventRecorder.Event(service, v1.EventTypeWarning, eventType, message)
		return err
	}
	// Always update the cache upon success.
	// NOTE: Since we update the cached service if and only if we successfully
	// processed it, a cached service being nil implies that it hasn't yet
	// been successfully processed.
	s.cache.set(key, cachedService)

	return nil
}

// createLoadBalancerIfNeeded ensures that service's status is synced up with loadbalancer
// i.e. creates loadbalancer for service if requested and deletes loadbalancer if the service
// doesn't want a loadbalancer no more. Returns whatever error occurred.
func (s *ServiceController) createLoadBalancerIfNeeded(key string, service *v1.Service) error {
	// Note: It is safe to just call EnsureLoadBalancer.  But, on some clouds that requires a delete & create,
	// which may involve service interruption.  Also, we would like user-friendly events.

	// Save the state so we can avoid a write if it doesn't change
	previousState := v1helper.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
	var newState *v1.LoadBalancerStatus
	var err error

	if !wantsLoadBalancer(service) {
		_, exists, err := s.balancer.GetLoadBalancer(context.TODO(), s.clusterName, service)
		if err != nil {
			return fmt.Errorf("error getting LB for service %s: %v", key, err)
		}
		if exists {
			glog.Infof("Deleting existing load balancer for service %s that no longer needs a load balancer.", key)
			s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
			if err := s.balancer.EnsureLoadBalancerDeleted(context.TODO(), s.clusterName, service); err != nil {
				return err
			}
			s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
		}

		newState = &v1.LoadBalancerStatus{}
	} else {
		glog.V(2).Infof("Ensuring LB for service %s", key)

		// TODO: We could do a dry-run here if wanted to avoid the spurious cloud-calls & events when we restart

		s.eventRecorder.Event(service, v1.EventTypeNormal, "EnsuringLoadBalancer", "Ensuring load balancer")
		newState, err = s.ensureLoadBalancer(service)
		if err != nil {
			return fmt.Errorf("failed to ensure load balancer for service %s: %v", key, err)
		}
		s.eventRecorder.Event(service, v1.EventTypeNormal, "EnsuredLoadBalancer", "Ensured load balancer")
	}

	// Write the state if changed
	// TODO: Be careful here ... what if there were other changes to the service?
	if !v1helper.LoadBalancerStatusEqual(previousState, newState) {
		// Make a copy so we don't mutate the shared informer cache
		service = service.DeepCopy()

		// Update the status on the copy
		service.Status.LoadBalancer = *newState

		if err := s.persistUpdate(service); err != nil {
			// TODO: This logic needs to be revisited. We might want to retry on all the errors, not just conflicts.
			if errors.IsConflict(err) {
				return fmt.Errorf("not persisting update to service '%s/%s' that has been changed since we received it: %v", service.Namespace, service.Name, err)
			}
			runtime.HandleError(fmt.Errorf("failed to persist service %q updated status to apiserver, even after retries. Giving up: %v", key, err))
			return nil
		}
	} else {
		glog.V(2).Infof("Not persisting unchanged LoadBalancerStatus for service %s to registry.", key)
	}

	return nil
}

func (s *ServiceController) persistUpdate(service *v1.Service) error {
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err = s.kubeClient.CoreV1().Services(service.Namespace).UpdateStatus(service)
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
		// balancer status. For now, just pass it up the stack.
		if errors.IsConflict(err) {
			return err
		}
		glog.Warningf("Failed to persist updated LoadBalancerStatus to service '%s/%s' after creating its load balancer: %v",
			service.Namespace, service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (s *ServiceController) ensureLoadBalancer(service *v1.Service) (*v1.LoadBalancerStatus, error) {
	nodes, err := s.nodeLister.ListWithPredicate(getNodeConditionPredicate())
	if err != nil {
		return nil, err
	}

	// If there are no available nodes for LoadBalancer service, make a EventTypeWarning event for it.
	if len(nodes) == 0 {
		s.eventRecorder.Eventf(service, v1.EventTypeWarning, "UnAvailableLoadBalancer", "There are no available nodes for LoadBalancer service %s/%s", service.Namespace, service.Name)
	}

	// - Only one protocol supported per service
	// - Not all cloud providers support all protocols and the next step is expected to return
	//   an error for unsupported protocols
	return s.balancer.EnsureLoadBalancer(context.TODO(), s.clusterName, service, nodes)
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
func (s *serviceCache) allServices() []*v1.Service {
	s.mu.Lock()
	defer s.mu.Unlock()
	services := make([]*v1.Service, 0, len(s.serviceMap))
	for _, v := range s.serviceMap {
		services = append(services, v.state)
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

func (s *ServiceController) needsUpdate(oldService *v1.Service, newService *v1.Service) bool {
	if !wantsLoadBalancer(oldService) && !wantsLoadBalancer(newService) {
		return false
	}
	if wantsLoadBalancer(oldService) != wantsLoadBalancer(newService) {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "Type", "%v -> %v",
			oldService.Spec.Type, newService.Spec.Type)
		return true
	}

	if wantsLoadBalancer(newService) && !reflect.DeepEqual(oldService.Spec.LoadBalancerSourceRanges, newService.Spec.LoadBalancerSourceRanges) {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "LoadBalancerSourceRanges", "%v -> %v",
			oldService.Spec.LoadBalancerSourceRanges, newService.Spec.LoadBalancerSourceRanges)
		return true
	}

	if !portsEqualForLB(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		return true
	}
	if !loadBalancerIPsAreEqual(oldService, newService) {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "LoadbalancerIP", "%v -> %v",
			oldService.Spec.LoadBalancerIP, newService.Spec.LoadBalancerIP)
		return true
	}
	if len(oldService.Spec.ExternalIPs) != len(newService.Spec.ExternalIPs) {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "ExternalIP", "Count: %v -> %v",
			len(oldService.Spec.ExternalIPs), len(newService.Spec.ExternalIPs))
		return true
	}
	for i := range oldService.Spec.ExternalIPs {
		if oldService.Spec.ExternalIPs[i] != newService.Spec.ExternalIPs[i] {
			s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "ExternalIP", "Added: %v",
				newService.Spec.ExternalIPs[i])
			return true
		}
	}
	if !reflect.DeepEqual(oldService.Annotations, newService.Annotations) {
		return true
	}
	if oldService.UID != newService.UID {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "UID", "%v -> %v",
			oldService.UID, newService.UID)
		return true
	}
	if oldService.Spec.ExternalTrafficPolicy != newService.Spec.ExternalTrafficPolicy {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "ExternalTrafficPolicy", "%v -> %v",
			oldService.Spec.ExternalTrafficPolicy, newService.Spec.ExternalTrafficPolicy)
		return true
	}
	if oldService.Spec.HealthCheckNodePort != newService.Spec.HealthCheckNodePort {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "HealthCheckNodePort", "%v -> %v",
			oldService.Spec.HealthCheckNodePort, newService.Spec.HealthCheckNodePort)
		return true
	}

	return false
}

func (s *ServiceController) loadBalancerName(service *v1.Service) string {
	return s.balancer.GetLoadBalancerName(context.TODO(), "", service)
}

func getPortsForLB(service *v1.Service) ([]*v1.ServicePort, error) {
	var protocol v1.Protocol

	ports := []*v1.ServicePort{}
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]
		// The check on protocol was removed here.  The cloud provider itself is now responsible for all protocol validation
		ports = append(ports, sp)
		if protocol == "" {
			protocol = sp.Protocol
		} else if protocol != sp.Protocol && wantsLoadBalancer(service) {
			// TODO:  Convert error messages to use event recorder
			return nil, fmt.Errorf("mixed protocol external load balancers are not supported")
		}
	}
	return ports, nil
}

func portsEqualForLB(x, y *v1.Service) bool {
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

func portSlicesEqualForLB(x, y []*v1.ServicePort) bool {
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

func portEqualForLB(x, y *v1.ServicePort) bool {
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

func nodeNames(nodes []*v1.Node) sets.String {
	ret := sets.NewString()
	for _, node := range nodes {
		ret.Insert(node.Name)
	}
	return ret
}

func nodeSlicesEqualForLB(x, y []*v1.Node) bool {
	if len(x) != len(y) {
		return false
	}
	return nodeNames(x).Equal(nodeNames(y))
}

func getNodeConditionPredicate() corelisters.NodeConditionPredicate {
	return func(node *v1.Node) bool {
		// We add the master to the node list, but its unschedulable.  So we use this to filter
		// the master.
		if node.Spec.Unschedulable {
			return false
		}

		// As of 1.6, we will taint the master, but not necessarily mark it unschedulable.
		// Recognize nodes labeled as master, and filter them also, as we were doing previously.
		if _, hasMasterRoleLabel := node.Labels[LabelNodeRoleMaster]; hasMasterRoleLabel {
			return false
		}

		if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.ServiceNodeExclusion) {
			if _, hasExcludeBalancerLabel := node.Labels[LabelNodeRoleExcludeBalancer]; hasExcludeBalancerLabel {
				return false
			}
		}

		// If we have no info, don't accept
		if len(node.Status.Conditions) == 0 {
			return false
		}
		for _, cond := range node.Status.Conditions {
			// We consider the node for load balancing only when its NodeReady condition status
			// is ConditionTrue
			if cond.Type == v1.NodeReady && cond.Status != v1.ConditionTrue {
				glog.V(4).Infof("Ignoring node %v with %v condition status %v", node.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// nodeSyncLoop handles updating the hosts pointed to by all load
// balancers whenever the set of nodes in the cluster changes.
func (s *ServiceController) nodeSyncLoop() {
	newHosts, err := s.nodeLister.ListWithPredicate(getNodeConditionPredicate())
	if err != nil {
		glog.Errorf("Failed to retrieve current set of nodes from node lister: %v", err)
		return
	}
	if nodeSlicesEqualForLB(newHosts, s.knownHosts) {
		// The set of nodes in the cluster hasn't changed, but we can retry
		// updating any services that we failed to update last time around.
		s.servicesToUpdate = s.updateLoadBalancerHosts(s.servicesToUpdate, newHosts)
		return
	}

	glog.Infof("Detected change in list of current cluster nodes. New node set: %v",
		nodeNames(newHosts))

	// Try updating all services, and save the ones that fail to try again next
	// round.
	s.servicesToUpdate = s.cache.allServices()
	numServices := len(s.servicesToUpdate)
	s.servicesToUpdate = s.updateLoadBalancerHosts(s.servicesToUpdate, newHosts)
	glog.Infof("Successfully updated %d out of %d load balancers to direct traffic to the updated set of nodes",
		numServices-len(s.servicesToUpdate), numServices)

	s.knownHosts = newHosts
}

// updateLoadBalancerHosts updates all existing load balancers so that
// they will match the list of hosts provided.
// Returns the list of services that couldn't be updated.
func (s *ServiceController) updateLoadBalancerHosts(services []*v1.Service, hosts []*v1.Node) (servicesToRetry []*v1.Service) {
	for _, service := range services {
		func() {
			if service == nil {
				return
			}
			if err := s.lockedUpdateLoadBalancerHosts(service, hosts); err != nil {
				glog.Errorf("External error while updating load balancer: %v.", err)
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// Updates the load balancer of a service, assuming we hold the mutex
// associated with the service.
func (s *ServiceController) lockedUpdateLoadBalancerHosts(service *v1.Service, hosts []*v1.Node) error {
	if !wantsLoadBalancer(service) {
		return nil
	}

	// This operation doesn't normally take very long (and happens pretty often), so we only record the final event
	err := s.balancer.UpdateLoadBalancer(context.TODO(), s.clusterName, service, hosts)
	if err == nil {
		// If there are no available nodes for LoadBalancer service, make a EventTypeWarning event for it.
		if len(hosts) == 0 {
			s.eventRecorder.Eventf(service, v1.EventTypeWarning, "UnAvailableLoadBalancer", "There are no available nodes for LoadBalancer service %s/%s", service.Namespace, service.Name)
		} else {
			s.eventRecorder.Event(service, v1.EventTypeNormal, "UpdatedLoadBalancer", "Updated load balancer with new hosts")
		}
		return nil
	}

	// It's only an actual error if the load balancer still exists.
	if _, exists, err := s.balancer.GetLoadBalancer(context.TODO(), s.clusterName, service); err != nil {
		glog.Errorf("External error while checking if load balancer %q exists: name, %v", s.balancer.GetLoadBalancerName(context.TODO(), s.clusterName, service), err)
	} else if !exists {
		return nil
	}

	s.eventRecorder.Eventf(service, v1.EventTypeWarning, "LoadBalancerUpdateFailed", "Error updating load balancer with new hosts %v: %v", nodeNames(hosts), err)
	return err
}

func wantsLoadBalancer(service *v1.Service) bool {
	return service.Spec.Type == v1.ServiceTypeLoadBalancer
}

func loadBalancerIPsAreEqual(oldService, newService *v1.Service) bool {
	return oldService.Spec.LoadBalancerIP == newService.Spec.LoadBalancerIP
}

// syncService will sync the Service with the given key if it has had its expectations fulfilled,
// meaning it did not expect to see any more of its pods created or deleted. This function is not meant to be
// invoked concurrently with the same key.
func (s *ServiceController) syncService(key string) error {
	startTime := time.Now()
	var cachedService *cachedService
	defer func() {
		glog.V(4).Infof("Finished syncing service %q (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	// service holds the latest service info from apiserver
	service, err := s.serviceLister.Services(namespace).Get(name)
	switch {
	case errors.IsNotFound(err):
		// service absence in store means watcher caught the deletion, ensure LB info is cleaned
		glog.Infof("Service has been deleted %v. Attempting to cleanup load balancer resources", key)
		err = s.processServiceDeletion(key)
	case err != nil:
		glog.Infof("Unable to retrieve service %v from store: %v", key, err)
	default:
		cachedService = s.cache.getOrCreate(key)
		err = s.processServiceUpdate(cachedService, service, key)
	}

	return err
}

// Returns an error if processing the service deletion failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry after that Duration.
func (s *ServiceController) processServiceDeletion(key string) error {
	cachedService, ok := s.cache.get(key)
	if !ok {
		glog.Errorf("service %s not in cache even though the watcher thought it was. Ignoring the deletion", key)
		return nil
	}
	return s.processLoadBalancerDelete(cachedService, key)
}

func (s *ServiceController) processLoadBalancerDelete(cachedService *cachedService, key string) error {
	service := cachedService.state
	// delete load balancer info only if the service type is LoadBalancer
	if !wantsLoadBalancer(service) {
		return nil
	}
	s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
	err := s.balancer.EnsureLoadBalancerDeleted(context.TODO(), s.clusterName, service)
	if err != nil {
		s.eventRecorder.Eventf(service, v1.EventTypeWarning, "DeletingLoadBalancerFailed", "Error deleting load balancer (will retry): %v", err)
		return err
	}
	s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
	s.cache.delete(key)

	return nil
}
