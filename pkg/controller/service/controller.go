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

	v1 "k8s.io/api/core/v1"
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
	cloudprovider "k8s.io/cloud-provider"
	servicehelper "k8s.io/cloud-provider/service/helpers"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog"
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

	// labelNodeRoleMaster specifies that a node is a master. The use of this label within the
	// controller is deprecated and only considered when the LegacyNodeRoleBehavior feature gate
	// is on.
	labelNodeRoleMaster = "node-role.kubernetes.io/master"

	// labelNodeRoleExcludeBalancer specifies that the node should not be considered as a target
	// for external load-balancers which use nodes as a second hop (e.g. many cloud LBs which only
	// understand nodes). For services that use externalTrafficPolicy=Local, this may mean that
	// any backends on excluded nodes are not reachable by those external load-balancers.
	// Implementations of this exclusion may vary based on provider. This label is honored starting
	// in 1.16 when the ServiceNodeExclusion gate is on.
	labelNodeRoleExcludeBalancer = "node.kubernetes.io/exclude-from-external-load-balancers"

	// labelAlphaNodeRoleExcludeBalancer specifies that the node should be
	// exclude from load balancers created by a cloud provider. This label is deprecated and will
	// be removed in 1.18.
	labelAlphaNodeRoleExcludeBalancer = "alpha.service-controller.kubernetes.io/exclude-balancer"

	// serviceNodeExclusionFeature is the feature gate name that
	// enables nodes to exclude themselves from service load balancers
	// originated from: https://github.com/kubernetes/kubernetes/blob/28e800245e/pkg/features/kube_features.go#L178
	serviceNodeExclusionFeature = "ServiceNodeExclusion"

	// legacyNodeRoleBehaviro is the feature gate name that enables legacy
	// behavior to vary cluster functionality on the node-role.kubernetes.io
	// labels.
	legacyNodeRoleBehaviorFeature = "LegacyNodeRoleBehavior"
)

type cachedService struct {
	// The cached state of the service
	state *v1.Service
}

type serviceCache struct {
	mu         sync.RWMutex // protects serviceMap
	serviceMap map[string]*cachedService
}

// Controller keeps cloud provider service resources
// (like load balancers) in sync with the registry.
type Controller struct {
	cloud            cloudprovider.Interface
	knownHosts       []*v1.Node
	servicesToUpdate []*v1.Service
	kubeClient       clientset.Interface
	clusterName      string
	balancer         cloudprovider.LoadBalancer
	// TODO(#85155): Stop relying on this and remove the cache completely.
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
) (*Controller, error) {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(klog.Infof)
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "service-controller"})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := ratelimiter.RegisterMetricAndTrackRateLimiterUsage("service_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	s := &Controller{
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
			AddFunc: func(cur interface{}) {
				svc, ok := cur.(*v1.Service)
				// Check cleanup here can provide a remedy when controller failed to handle
				// changes before it exiting (e.g. crashing, restart, etc.).
				if ok && (wantsLoadBalancer(svc) || needsCleanup(svc)) {
					s.enqueueService(cur)
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				oldSvc, ok1 := old.(*v1.Service)
				curSvc, ok2 := cur.(*v1.Service)
				if ok1 && ok2 && (s.needsUpdate(oldSvc, curSvc) || needsCleanup(curSvc)) {
					s.enqueueService(cur)
				}
			},
			// No need to handle deletion event because the deletion would be handled by
			// the update path when the deletion timestamp is added.
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
func (s *Controller) enqueueService(obj interface{}) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		runtime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
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
func (s *Controller) Run(stopCh <-chan struct{}, workers int) {
	defer runtime.HandleCrash()
	defer s.queue.ShutDown()

	klog.Info("Starting service controller")
	defer klog.Info("Shutting down service controller")

	if !cache.WaitForNamedCacheSync("service", stopCh, s.serviceListerSynced, s.nodeListerSynced) {
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
func (s *Controller) worker() {
	for s.processNextWorkItem() {
	}
}

func (s *Controller) processNextWorkItem() bool {
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

func (s *Controller) init() error {
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

// processServiceCreateOrUpdate operates loadbalancers for the incoming service accordingly.
// Returns an error if processing the service update failed.
func (s *Controller) processServiceCreateOrUpdate(service *v1.Service, key string) error {
	// TODO(@MrHohn): Remove the cache once we get rid of the non-finalizer deletion
	// path. Ref https://github.com/kubernetes/enhancements/issues/980.
	cachedService := s.cache.getOrCreate(key)
	if cachedService.state != nil && cachedService.state.UID != service.UID {
		// This happens only when a service is deleted and re-created
		// in a short period, which is only possible when it doesn't
		// contain finalizer.
		if err := s.processLoadBalancerDelete(cachedService.state, key); err != nil {
			return err
		}
	}
	// Always cache the service, we need the info for service deletion in case
	// when load balancer cleanup is not handled via finalizer.
	cachedService.state = service
	op, err := s.syncLoadBalancerIfNeeded(service, key)
	if err != nil {
		s.eventRecorder.Eventf(service, v1.EventTypeWarning, "SyncLoadBalancerFailed", "Error syncing load balancer: %v", err)
		return err
	}
	if op == deleteLoadBalancer {
		// Only delete the cache upon successful load balancer deletion.
		s.cache.delete(key)
	}

	return nil
}

type loadBalancerOperation int

const (
	deleteLoadBalancer loadBalancerOperation = iota
	ensureLoadBalancer
)

// syncLoadBalancerIfNeeded ensures that service's status is synced up with loadbalancer
// i.e. creates loadbalancer for service if requested and deletes loadbalancer if the service
// doesn't want a loadbalancer no more. Returns whatever error occurred.
func (s *Controller) syncLoadBalancerIfNeeded(service *v1.Service, key string) (loadBalancerOperation, error) {
	// Note: It is safe to just call EnsureLoadBalancer.  But, on some clouds that requires a delete & create,
	// which may involve service interruption.  Also, we would like user-friendly events.

	// Save the state so we can avoid a write if it doesn't change
	previousStatus := service.Status.LoadBalancer.DeepCopy()
	var newStatus *v1.LoadBalancerStatus
	var op loadBalancerOperation
	var err error

	if !wantsLoadBalancer(service) || needsCleanup(service) {
		// Delete the load balancer if service no longer wants one, or if service needs cleanup.
		op = deleteLoadBalancer
		newStatus = &v1.LoadBalancerStatus{}
		_, exists, err := s.balancer.GetLoadBalancer(context.TODO(), s.clusterName, service)
		if err != nil {
			return op, fmt.Errorf("failed to check if load balancer exists before cleanup: %v", err)
		}
		if exists {
			klog.V(2).Infof("Deleting existing load balancer for service %s", key)
			s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
			if err := s.balancer.EnsureLoadBalancerDeleted(context.TODO(), s.clusterName, service); err != nil {
				return op, fmt.Errorf("failed to delete load balancer: %v", err)
			}
		}
		// Always remove finalizer when load balancer is deleted, this ensures Services
		// can be deleted after all corresponding load balancer resources are deleted.
		if err := s.removeFinalizer(service); err != nil {
			return op, fmt.Errorf("failed to remove load balancer cleanup finalizer: %v", err)
		}
		s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
	} else {
		// Create or update the load balancer if service wants one.
		op = ensureLoadBalancer
		klog.V(2).Infof("Ensuring load balancer for service %s", key)
		s.eventRecorder.Event(service, v1.EventTypeNormal, "EnsuringLoadBalancer", "Ensuring load balancer")
		// Always add a finalizer prior to creating load balancers, this ensures Services
		// can't be deleted until all corresponding load balancer resources are also deleted.
		if err := s.addFinalizer(service); err != nil {
			return op, fmt.Errorf("failed to add load balancer cleanup finalizer: %v", err)
		}
		newStatus, err = s.ensureLoadBalancer(service)
		if err != nil {
			if err == cloudprovider.ImplementedElsewhere {
				// ImplementedElsewhere indicates that the ensureLoadBalancer is a nop and the
				// functionality is implemented by a different controller.  In this case, we
				// return immediately without doing anything.
				klog.V(4).Infof("LoadBalancer for service %s implemented by a different controller %s, Ignoring error", key, s.cloud.ProviderName())
				return op, nil
			}
			return op, fmt.Errorf("failed to ensure load balancer: %v", err)
		}
		s.eventRecorder.Event(service, v1.EventTypeNormal, "EnsuredLoadBalancer", "Ensured load balancer")
	}

	if err := s.patchStatus(service, previousStatus, newStatus); err != nil {
		// Only retry error that isn't not found:
		// - Not found error mostly happens when service disappears right after
		//   we remove the finalizer.
		// - We can't patch status on non-exist service anyway.
		if !errors.IsNotFound(err) {
			return op, fmt.Errorf("failed to update load balancer status: %v", err)
		}
	}

	return op, nil
}

func (s *Controller) ensureLoadBalancer(service *v1.Service) (*v1.LoadBalancerStatus, error) {
	nodes, err := s.nodeLister.ListWithPredicate(getNodeConditionPredicate())
	if err != nil {
		return nil, err
	}

	// If there are no available nodes for LoadBalancer service, make a EventTypeWarning event for it.
	if len(nodes) == 0 {
		s.eventRecorder.Event(service, v1.EventTypeWarning, "UnAvailableLoadBalancer", "There are no available nodes for LoadBalancer")
	}

	// - Only one protocol supported per service
	// - Not all cloud providers support all protocols and the next step is expected to return
	//   an error for unsupported protocols
	return s.balancer.EnsureLoadBalancer(context.TODO(), s.clusterName, service, nodes)
}

// ListKeys implements the interface required by DeltaFIFO to list the keys we
// already know about.
func (s *serviceCache) ListKeys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	keys := make([]string, 0, len(s.serviceMap))
	for k := range s.serviceMap {
		keys = append(keys, k)
	}
	return keys
}

// GetByKey returns the value stored in the serviceMap under the given key
func (s *serviceCache) GetByKey(key string) (interface{}, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if v, ok := s.serviceMap[key]; ok {
		return v, true, nil
	}
	return nil, false, nil
}

// ListKeys implements the interface required by DeltaFIFO to list the keys we
// already know about.
func (s *serviceCache) allServices() []*v1.Service {
	s.mu.RLock()
	defer s.mu.RUnlock()
	services := make([]*v1.Service, 0, len(s.serviceMap))
	for _, v := range s.serviceMap {
		services = append(services, v.state)
	}
	return services
}

func (s *serviceCache) get(serviceName string) (*cachedService, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
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

// needsCleanup checks if load balancer needs to be cleaned up as indicated by finalizer.
func needsCleanup(service *v1.Service) bool {
	if !servicehelper.HasLBFinalizer(service) {
		return false
	}

	if service.ObjectMeta.DeletionTimestamp != nil {
		return true
	}

	// Service doesn't want loadBalancer but owns loadBalancer finalizer also need to be cleaned up.
	if service.Spec.Type != v1.ServiceTypeLoadBalancer {
		return true
	}

	return false
}

// needsUpdate checks if load balancer needs to be updated due to change in attributes.
func (s *Controller) needsUpdate(oldService *v1.Service, newService *v1.Service) bool {
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

	if !reflect.DeepEqual(oldService.Spec.SessionAffinityConfig, newService.Spec.SessionAffinityConfig) {
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

func getPortsForLB(service *v1.Service) []*v1.ServicePort {
	ports := []*v1.ServicePort{}
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]
		ports = append(ports, sp)
	}
	return ports
}

func portsEqualForLB(x, y *v1.Service) bool {
	xPorts := getPortsForLB(x)
	yPorts := getPortsForLB(y)
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

	if x.TargetPort != y.TargetPort {
		return false
	}

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

		if utilfeature.DefaultFeatureGate.Enabled(legacyNodeRoleBehaviorFeature) {
			// As of 1.6, we will taint the master, but not necessarily mark it unschedulable.
			// Recognize nodes labeled as master, and filter them also, as we were doing previously.
			if _, hasMasterRoleLabel := node.Labels[labelNodeRoleMaster]; hasMasterRoleLabel {
				return false
			}
		}
		if utilfeature.DefaultFeatureGate.Enabled(serviceNodeExclusionFeature) {
			// Will be removed in 1.18
			if _, hasExcludeBalancerLabel := node.Labels[labelAlphaNodeRoleExcludeBalancer]; hasExcludeBalancerLabel {
				return false
			}
			if _, hasExcludeBalancerLabel := node.Labels[labelNodeRoleExcludeBalancer]; hasExcludeBalancerLabel {
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
				klog.V(4).Infof("Ignoring node %v with %v condition status %v", node.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// nodeSyncLoop handles updating the hosts pointed to by all load
// balancers whenever the set of nodes in the cluster changes.
func (s *Controller) nodeSyncLoop() {
	newHosts, err := s.nodeLister.ListWithPredicate(getNodeConditionPredicate())
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to retrieve current set of nodes from node lister: %v", err))
		return
	}
	if nodeSlicesEqualForLB(newHosts, s.knownHosts) {
		// The set of nodes in the cluster hasn't changed, but we can retry
		// updating any services that we failed to update last time around.
		s.servicesToUpdate = s.updateLoadBalancerHosts(s.servicesToUpdate, newHosts)
		return
	}

	klog.V(2).Infof("Detected change in list of current cluster nodes. New node set: %v",
		nodeNames(newHosts))

	// Try updating all services, and save the ones that fail to try again next
	// round.
	s.servicesToUpdate = s.cache.allServices()
	numServices := len(s.servicesToUpdate)
	s.servicesToUpdate = s.updateLoadBalancerHosts(s.servicesToUpdate, newHosts)
	klog.V(2).Infof("Successfully updated %d out of %d load balancers to direct traffic to the updated set of nodes",
		numServices-len(s.servicesToUpdate), numServices)

	s.knownHosts = newHosts
}

// updateLoadBalancerHosts updates all existing load balancers so that
// they will match the list of hosts provided.
// Returns the list of services that couldn't be updated.
func (s *Controller) updateLoadBalancerHosts(services []*v1.Service, hosts []*v1.Node) (servicesToRetry []*v1.Service) {
	for _, service := range services {
		func() {
			if service == nil {
				return
			}
			if err := s.lockedUpdateLoadBalancerHosts(service, hosts); err != nil {
				runtime.HandleError(fmt.Errorf("failed to update load balancer hosts for service %s/%s: %v", service.Namespace, service.Name, err))
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// Updates the load balancer of a service, assuming we hold the mutex
// associated with the service.
func (s *Controller) lockedUpdateLoadBalancerHosts(service *v1.Service, hosts []*v1.Node) error {
	if !wantsLoadBalancer(service) {
		return nil
	}

	// This operation doesn't normally take very long (and happens pretty often), so we only record the final event
	err := s.balancer.UpdateLoadBalancer(context.TODO(), s.clusterName, service, hosts)
	if err == nil {
		// If there are no available nodes for LoadBalancer service, make a EventTypeWarning event for it.
		if len(hosts) == 0 {
			s.eventRecorder.Event(service, v1.EventTypeWarning, "UnAvailableLoadBalancer", "There are no available nodes for LoadBalancer")
		} else {
			s.eventRecorder.Event(service, v1.EventTypeNormal, "UpdatedLoadBalancer", "Updated load balancer with new hosts")
		}
		return nil
	}
	if err == cloudprovider.ImplementedElsewhere {
		// ImplementedElsewhere indicates that the UpdateLoadBalancer is a nop and the
		// functionality is implemented by a different controller.  In this case, we
		// return immediately without doing anything.
		return nil
	}
	// It's only an actual error if the load balancer still exists.
	if _, exists, err := s.balancer.GetLoadBalancer(context.TODO(), s.clusterName, service); err != nil {
		runtime.HandleError(fmt.Errorf("failed to check if load balancer exists for service %s/%s: %v", service.Namespace, service.Name, err))
	} else if !exists {
		return nil
	}

	s.eventRecorder.Eventf(service, v1.EventTypeWarning, "UpdateLoadBalancerFailed", "Error updating load balancer with new hosts %v: %v", nodeNames(hosts), err)
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
func (s *Controller) syncService(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing service %q (%v)", key, time.Since(startTime))
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
		err = s.processServiceDeletion(key)
	case err != nil:
		runtime.HandleError(fmt.Errorf("Unable to retrieve service %v from store: %v", key, err))
	default:
		err = s.processServiceCreateOrUpdate(service, key)
	}

	return err
}

func (s *Controller) processServiceDeletion(key string) error {
	cachedService, ok := s.cache.get(key)
	if !ok {
		// Cache does not contains the key means:
		// - We didn't create a Load Balancer for the deleted service at all.
		// - We already deleted the Load Balancer that was created for the service.
		// In both cases we have nothing left to do.
		return nil
	}
	klog.V(2).Infof("Service %v has been deleted. Attempting to cleanup load balancer resources", key)
	if err := s.processLoadBalancerDelete(cachedService.state, key); err != nil {
		return err
	}
	s.cache.delete(key)
	return nil
}

func (s *Controller) processLoadBalancerDelete(service *v1.Service, key string) error {
	// delete load balancer info only if the service type is LoadBalancer
	if !wantsLoadBalancer(service) {
		return nil
	}
	s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletingLoadBalancer", "Deleting load balancer")
	if err := s.balancer.EnsureLoadBalancerDeleted(context.TODO(), s.clusterName, service); err != nil {
		s.eventRecorder.Eventf(service, v1.EventTypeWarning, "DeleteLoadBalancerFailed", "Error deleting load balancer: %v", err)
		return err
	}
	s.eventRecorder.Event(service, v1.EventTypeNormal, "DeletedLoadBalancer", "Deleted load balancer")
	return nil
}

// addFinalizer patches the service to add finalizer.
func (s *Controller) addFinalizer(service *v1.Service) error {
	if servicehelper.HasLBFinalizer(service) {
		return nil
	}

	// Make a copy so we don't mutate the shared informer cache.
	updated := service.DeepCopy()
	updated.ObjectMeta.Finalizers = append(updated.ObjectMeta.Finalizers, servicehelper.LoadBalancerCleanupFinalizer)

	klog.V(2).Infof("Adding finalizer to service %s/%s", updated.Namespace, updated.Name)
	_, err := patch(s.kubeClient.CoreV1(), service, updated)
	return err
}

// removeFinalizer patches the service to remove finalizer.
func (s *Controller) removeFinalizer(service *v1.Service) error {
	if !servicehelper.HasLBFinalizer(service) {
		return nil
	}

	// Make a copy so we don't mutate the shared informer cache.
	updated := service.DeepCopy()
	updated.ObjectMeta.Finalizers = removeString(updated.ObjectMeta.Finalizers, servicehelper.LoadBalancerCleanupFinalizer)

	klog.V(2).Infof("Removing finalizer from service %s/%s", updated.Namespace, updated.Name)
	_, err := patch(s.kubeClient.CoreV1(), service, updated)
	return err
}

// removeString returns a newly created []string that contains all items from slice that
// are not equal to s.
func removeString(slice []string, s string) []string {
	var newSlice []string
	for _, item := range slice {
		if item != s {
			newSlice = append(newSlice, item)
		}
	}
	return newSlice
}

// patchStatus patches the service with the given LoadBalancerStatus.
func (s *Controller) patchStatus(service *v1.Service, previousStatus, newStatus *v1.LoadBalancerStatus) error {
	if servicehelper.LoadBalancerStatusEqual(previousStatus, newStatus) {
		return nil
	}
	// Make a copy so we don't mutate the shared informer cache.
	updated := service.DeepCopy()
	updated.Status.LoadBalancer = *newStatus

	klog.V(2).Infof("Patching status for service %s/%s", updated.Namespace, updated.Name)
	_, err := patch(s.kubeClient.CoreV1(), service, updated)
	return err
}
