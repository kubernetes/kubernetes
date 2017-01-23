/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"time"

	"github.com/golang/glog"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/pkg/util/flowcontrol"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	si "k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/legacylisters"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

const (
	UserAgentName = "federation-service-controller"

	reviewDelay           = 10 * time.Second
	smallDelay            = 3 * time.Second
	updateTimeout         = 30 * time.Second
	clusterAvailableDelay = 20 * time.Second
	allClustersKey        = "ALL_CLUSTERS"
)

type ServiceController struct {
	federationClient      fedclientset.Interface
	reviewDelay           time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
	// services that should be federated.
	serviceStore listers.StoreToServiceLister
	// Informer controller for services that should be federated.
	serviceController cache.Controller
	// Contains services present in members of federation.
	federatedInformer fedutil.FederatedInformer
	// Contains endpoints present in members of federation.
	endpointFederatedInformer fedutil.FederatedInformer
	eventRecorder             record.EventRecorder
	// services that need to be synced
	queue *workqueue.Type
	// For triggering single service reconciliation. This is used when there is an add/update/delete operation
	// on a service in either federated API server or in some member of the federation.
	objectDeliverer *fedutil.DelayingDeliverer
	// For triggering all services reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *fedutil.DelayingDeliverer

	deletionHelper *deletionhelper.DeletionHelper
	// For updating members of federation.
	federatedUpdater fedutil.FederatedUpdater
	// ServiceBackoff manager
	flowcontrolBackoff *flowcontrol.Backoff
}

// New returns a new service controller
func New(federationClient fedclientset.Interface) *ServiceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(federationClient))
	recorder := broadcaster.NewRecorder(v1.EventSource{Component: UserAgentName})

	s := &ServiceController{
		federationClient:      federationClient,
		eventRecorder:         recorder,
		queue:                 workqueue.New(),
		reviewDelay:           reviewDelay,
		clusterAvailableDelay: clusterAvailableDelay,
		smallDelay:            smallDelay,
		updateTimeout:         updateTimeout,
		flowcontrolBackoff:    flowcontrol.NewBackOff(5*time.Second, time.Minute),
	}

	// Build deliverers for triggering reconciliations.
	s.objectDeliverer = fedutil.NewDelayingDeliverer()
	s.clusterDeliverer = fedutil.NewDelayingDeliverer()

	// Start informer in federated API servers on services that should be federated.
	s.serviceStore.Indexer, s.serviceController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (pkgruntime.Object, error) {
				return s.federationClient.Core().Services(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return s.federationClient.Core().Services(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.Service{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnAllChanges(func(obj pkgruntime.Object) {
			glog.V(5).Infof("Delivering notification from federation: %v", obj)
			s.deliverServiceObj(obj, 0, false)
		}),
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	// Federated informer on services in members of federation.
	clusterLifecycle := fedutil.ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *v1beta1.Cluster) {
			// When new cluster becomes available, process all the services again.
			s.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
		},
		ClusterUnregistered: func(cluster *v1beta1.Cluster, _ []interface{}) {
			// When cluster is unregistered, process all the services again.
			s.clusterDeliverer.DeliverAfter(allClustersKey, nil, 0)
		},
	}
	fedInformerFactory := func(cluster *v1beta1.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options v1.ListOptions) (pkgruntime.Object, error) {
					return targetClient.Core().Services(v1.NamespaceAll).List(options)
				},
				WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
					return targetClient.Core().Services(v1.NamespaceAll).Watch(options)
				},
			},
			&v1.Service{},
			controller.NoResyncPeriodFunc(),
			// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
			// would be just confirmation that some service operation succeeded.
			fedutil.NewTriggerOnAllChanges(
				func(obj pkgruntime.Object) {
					glog.V(5).Infof("Delivering service notification from federated cluster %s: %v", cluster.Name, obj)
					s.deliverServiceObj(obj, s.reviewDelay, false)
				},
			))
	}

	s.federatedInformer = fedutil.NewFederatedInformer(federationClient, fedInformerFactory, &clusterLifecycle)

	s.federatedUpdater = fedutil.NewFederatedUpdater(s.federatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			svc := obj.(*v1.Service)
			_, err := client.Core().Services(svc.Namespace).Create(svc)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			svc := obj.(*v1.Service)
			_, err := client.Core().Services(svc.Namespace).Update(svc)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			svc := obj.(*v1.Service)
			err := client.Core().Services(svc.Namespace).Delete(svc.Name, &v1.DeleteOptions{})
			// IsNotFound error is fine since that means the object is deleted already.
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		})

	// Federated informer on endpoints in members of federation.
	// This will enable to check if service ingress endpoints are reachable
	s.endpointFederatedInformer = fedutil.NewFederatedInformer(
		federationClient,
		func(cluster *v1beta1.Cluster, targetClient kubeclientset.Interface) (
			cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options v1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Core().Endpoints(v1.NamespaceAll).List(options)
					},
					WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Endpoints(v1.NamespaceAll).Watch(options)
					},
				},
				&v1.Endpoints{},
				controller.NoResyncPeriodFunc(),
				fedutil.NewTriggerOnMetaAndFieldChanges(
					"Subsets",
					func(obj pkgruntime.Object) {
						glog.V(5).Infof("Delivering endpoint notification from federated cluster %s :%v", cluster.Name, obj)
						s.deliverEndpointObj(obj, s.reviewDelay, false)
					},
				))
		},
		&fedutil.ClusterLifecycleHandlerFuncs{},
	)

	s.deletionHelper = deletionhelper.NewDeletionHelper(
		s.hasFinalizerFunc,
		s.removeFinalizerFunc,
		s.addFinalizerFunc,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			service := obj.(*v1.Service)
			return service.Name
		},
		updateTimeout,
		s.eventRecorder,
		s.federatedInformer,
		s.federatedUpdater,
	)
	return s
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (s *ServiceController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	service := obj.(*v1.Service)
	for i := range service.ObjectMeta.Finalizers {
		if string(service.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a service.
func (s *ServiceController) removeFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	service := obj.(*v1.Service)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range service.ObjectMeta.Finalizers {
		if string(service.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, service.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	service.ObjectMeta.Finalizers = newFinalizers
	service, err := s.federationClient.Core().Services(service.Namespace).Update(service)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from service %s: %v", finalizer, service.Name, err)
	}
	return service, nil
}

// Adds the given finalizer to the given objects ObjectMeta.
// Assumes that the given object is a service.
func (s *ServiceController) addFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	service := obj.(*v1.Service)
	service.ObjectMeta.Finalizers = append(service.ObjectMeta.Finalizers, finalizer)
	service, err := s.federationClient.Core().Services(service.Namespace).Update(service)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizer %s to service %s: %v", finalizer, service.Name, err)
	}
	return service, nil
}

// obj could be an *api.Service, or a DeletionFinalStateUnknown marker item.
func (s *ServiceController) enqueueService(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	s.queue.Add(key)
}

// Run runs the informer controllers
func (s *ServiceController) Run(workers int, stopCh <-chan struct{}) {
	s.objectDeliverer.StartWithHandler(func(item *fedutil.DelayingDelivererItem) {
		s.queue.Add(item.Value.(*types.NamespacedName))
	})
	s.clusterDeliverer.StartWithHandler(func(_ *fedutil.DelayingDelivererItem) {
		s.deliverServicesOnClusterChange()
	})

	for i := 0; i < workers; i++ {
		go wait.Until(s.fedServiceWorker, time.Second, stopCh)
	}

	fedutil.StartBackoffGC(s.flowcontrolBackoff, stopCh)

	go s.serviceController.Run(stopCh)
	s.federatedInformer.Start()
	s.endpointFederatedInformer.Start()
	go func() {
		<-stopCh
		glog.Infof("Shutting down Federation Service Controller")
		s.federatedInformer.Stop()
		s.endpointFederatedInformer.Stop()
		s.objectDeliverer.Stop()
		s.clusterDeliverer.Stop()
		s.queue.ShutDown()
	}()
}

func (s *ServiceController) init() error {
	if s.federationName == "" {
		return fmt.Errorf("ServiceController should not be run without federationName.")
	}
	if s.zoneName == "" && s.zoneID == "" {
		return fmt.Errorf("ServiceController must be run with either zoneName or zoneID.")
	}
	if s.serviceDnsSuffix == "" {
		// TODO: Is this the right place to do defaulting?
		if s.zoneName == "" {
			return fmt.Errorf("ServiceController must be run with zoneName, if serviceDnsSuffix is not set.")
		}
		s.serviceDnsSuffix = s.zoneName
	}
	if s.dns == nil {
		return fmt.Errorf("ServiceController should not be run without a dnsprovider.")
	}
	zones, ok := s.dns.Zones()
	if !ok {
		return fmt.Errorf("the dns provider does not support zone enumeration, which is required for creating dns records.")
	}
	s.dnsZones = zones
	matchingZones, err := getDnsZones(s.zoneName, s.zoneID, s.dnsZones)
	if err != nil {
		return fmt.Errorf("error querying for DNS zones: %v", err)
	}
	if len(matchingZones) == 0 {
		if s.zoneName == "" {
			return fmt.Errorf("ServiceController must be run with zoneName to create zone automatically.")
		}
		glog.Infof("DNS zone %q not found.  Creating DNS zone %q.", s.zoneName, s.zoneName)
		managedZone, err := s.dnsZones.New(s.zoneName)
		if err != nil {
			return err
		}
		zone, err := s.dnsZones.Add(managedZone)
		if err != nil {
			return err
		}
		glog.Infof("DNS zone %q successfully created.  Note that DNS resolution will not work until you have registered this name with "+
			"a DNS registrar and they have changed the authoritative name servers for your domain to point to your DNS provider.", zone.Name())
	}
	if len(matchingZones) > 1 {
		return fmt.Errorf("Multiple matching DNS zones found for %q; please specify zoneID", s.zoneName)
	}
	return nil
}

type reconciliationStatus string

const (
	statusAllOk       = reconciliationStatus("ALL_OK")
	statusNeedRecheck = reconciliationStatus("RECHECK")
	statusError       = reconciliationStatus("ERROR")
	statusNotSynced   = reconciliationStatus("NOSYNC")
)

// fedServiceWorker runs a worker thread that just dequeues items, processes them, and marks them done.
func (s *ServiceController) fedServiceWorker() {
	for {
		key, quit := s.queue.Get()
		if quit {
			return
		}
		serviceName := *key.(*types.NamespacedName)
		status, err := s.reconcileService(serviceName)
		s.queue.Done(key)
		if err != nil {
			glog.Errorf("Error reconciling service: %v, delivering again", err)
			s.deliverObject(serviceName, 0, true)
		} else {
			switch status {
			case statusAllOk:
				break
			case statusError:
				glog.V(5).Infof("Delivering notification again upon error %v:", serviceName)
				s.deliverObject(serviceName, 0, true)
			case statusNeedRecheck:
				glog.V(5).Infof("Delivering notification again for recheck %v:", serviceName)
				s.deliverObject(serviceName, reviewDelay, false)
			case statusNotSynced:
				glog.V(5).Infof("Delivering notification after clusterAvailableDelay %v:", serviceName)
				s.deliverObject(serviceName, clusterAvailableDelay, false)
			default:
				glog.Errorf("Unhandled reconciliation status: %s, delivering again", status)
				s.deliverObject(serviceName, reviewDelay, false)
			}
		}
	}
}

func wantsDNSRecords(service *v1.Service) bool {
	return service.Spec.Type == v1.ServiceTypeLoadBalancer
}

// processServiceForCluster creates or updates service to all registered running clusters,
// update DNS records and update the service info with DNS entries to federation apiserver.
// the function returns any error caught
func (s *ServiceController) processServiceForCluster(cachedService *cachedService, clusterName string, service *v1.Service, client *kubeclientset.Clientset) error {
	glog.V(4).Infof("Process service %s/%s for cluster %s", service.Namespace, service.Name, clusterName)
	// Create or Update k8s Service
	err := s.ensureClusterService(cachedService, clusterName, service, client)
	if err != nil {
		glog.V(4).Infof("Failed to process service %s/%s for cluster %s", service.Namespace, service.Name, clusterName)
		return err
	}
	glog.V(4).Infof("Successfully process service %s/%s for cluster %s", service.Namespace, service.Name, clusterName)
	return nil
}

// updateFederationService Returns whatever error occurred along with a boolean indicator of whether it
// should be retried.
func (s *ServiceController) updateFederationService(key string, cachedService *cachedService) (error, bool) {
	// Clone federation service, and create them in underlying k8s cluster
	desiredService := &v1.Service{
		ObjectMeta: util.DeepCopyRelevantObjectMeta(cachedService.lastState.ObjectMeta),
		Spec:       *(util.DeepCopyApiTypeOrPanic(&cachedService.lastState.Spec).(*v1.ServiceSpec)),
	}

	// handle available clusters one by one
	var hasErr bool
	for clusterName, cache := range s.clusterCache.clientMap {
		go func(cache *clusterCache, clusterName string) {
			err := s.processServiceForCluster(cachedService, clusterName, desiredService, cache.clientset)
			if err != nil {
				hasErr = true
			}
		}(cache, clusterName)
	}
	if hasErr {
		// detail error has been dumped inside the loop
		return fmt.Errorf("Service %s/%s was not successfully updated to all clusters", desiredService.Namespace, desiredService.Name), retryable
	}
	return nil, !retryable
}

func (s *ServiceController) ensureClusterService(cachedService *cachedService, clusterName string, service *v1.Service, client *kubeclientset.Clientset) error {
	var err error
	var needUpdate bool
	for i := 0; i < clientRetryCount; i++ {
		svc, err := client.Core().Services(service.Namespace).Get(service.Name, metav1.GetOptions{})
		if err == nil {
			// service exists
			glog.V(5).Infof("Found service %s/%s from cluster %s", service.Namespace, service.Name, clusterName)
			//reserve immutable fields
			service.Spec.ClusterIP = svc.Spec.ClusterIP

			//reserve auto assigned field
			for i, oldPort := range svc.Spec.Ports {
				for _, port := range service.Spec.Ports {
					if port.NodePort == 0 {
						if !portEqualExcludeNodePort(&oldPort, &port) {
							svc.Spec.Ports[i] = port
							needUpdate = true
						}
					} else {
						if !portEqualForLB(&oldPort, &port) {
							svc.Spec.Ports[i] = port
							needUpdate = true
						}
					}
				}
			}

			if needUpdate {
				// we only apply spec update
				svc.Spec = service.Spec
				_, err = client.Core().Services(svc.Namespace).Update(svc)
				if err == nil {
					glog.V(5).Infof("Service %s/%s successfully updated to cluster %s", svc.Namespace, svc.Name, clusterName)
					return nil
				} else {
					glog.V(4).Infof("Failed to update %+v", err)
				}
			} else {
				glog.V(5).Infof("Service %s/%s is not updated to cluster %s as the spec are identical", svc.Namespace, svc.Name, clusterName)
				return nil
			}
		} else if errors.IsNotFound(err) {
			// Create service if it is not found
			glog.Infof("Service '%s/%s' is not found in cluster %s, trying to create new",
				service.Namespace, service.Name, clusterName)
			service.ResourceVersion = ""
			_, err = client.Core().Services(service.Namespace).Create(service)
			if err == nil {
				glog.V(5).Infof("Service %s/%s successfully created to cluster %s", service.Namespace, service.Name, clusterName)
				return nil
			}
			glog.V(4).Infof("Failed to create %+v", err)
			if errors.IsAlreadyExists(err) {
				glog.V(5).Infof("service %s/%s already exists in cluster %s", service.Namespace, service.Name, clusterName)
				return nil
			}
		}
		if errors.IsConflict(err) {
			glog.V(4).Infof("Not persisting update to service '%s/%s' that has been changed since we received it: %v",
				service.Namespace, service.Name, err)
		}
		// should we reuse same retry delay for all clusters?
		time.Sleep(cachedService.nextRetryDelay())
	}
	return err
}

func (s *serviceCache) allServices() []*cachedService {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	services := make([]*cachedService, 0, len(s.fedServiceMap))
	for _, v := range s.fedServiceMap {
		services = append(services, v)
	}
	return services
}

func (s *serviceCache) get(serviceName string) (*cachedService, bool) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	service, ok := s.fedServiceMap[serviceName]
	return service, ok
}

func (s *serviceCache) getOrCreate(serviceName string) *cachedService {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	service, ok := s.fedServiceMap[serviceName]
	if !ok {
		service = &cachedService{
			endpointMap:      make(map[string]int),
			serviceStatusMap: make(map[string]v1.LoadBalancerStatus),
		}
		s.fedServiceMap[serviceName] = service
	}
	return service
}

func (s *serviceCache) set(serviceName string, service *cachedService) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	s.fedServiceMap[serviceName] = service
}

func (s *serviceCache) delete(serviceName string) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	delete(s.fedServiceMap, serviceName)
}

// needsUpdateDNS check if the dns records of the given service should be updated
func (s *ServiceController) needsUpdateDNS(oldService *v1.Service, newService *v1.Service) bool {
	if !wantsDNSRecords(oldService) && !wantsDNSRecords(newService) {
		return false
	}
	if wantsDNSRecords(oldService) != wantsDNSRecords(newService) {
		s.eventRecorder.Eventf(newService, v1.EventTypeNormal, "Type", "%v -> %v",
			oldService.Spec.Type, newService.Spec.Type)
		return true
	}
	if !portsEqualForLB(oldService, newService) || oldService.Spec.SessionAffinity != newService.Spec.SessionAffinity {
		return true
	}
	if !LoadBalancerIPsAreEqual(oldService, newService) {
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

	return false
}

func getPortsForLB(service *v1.Service) ([]*v1.ServicePort, error) {
	// TODO: quinton: Probably applies for DNS SVC records.  Come back to this.
	//var protocol api.Protocol

	ports := []*v1.ServicePort{}
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]
		// The check on protocol was removed here.  The DNS provider itself is now responsible for all protocol validation
		ports = append(ports, sp)
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
	return true
}

func portEqualExcludeNodePort(x, y *v1.ServicePort) bool {
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
	return true
}

func clustersFromList(list *v1beta1.ClusterList) []string {
	result := []string{}
	for ix := range list.Items {
		result = append(result, list.Items[ix].Name)
	}
	return result
}

// getClusterConditionPredicate filter all clusters meet condition of
// condition.type=Ready and condition.status=true
func getClusterConditionPredicate() federationcache.ClusterConditionPredicate {
	return func(cluster v1beta1.Cluster) bool {
		// If we have no info, don't accept
		if len(cluster.Status.Conditions) == 0 {
			return false
		}
		for _, cond := range cluster.Status.Conditions {
			//We consider the cluster for load balancing only when its ClusterReady condition status
			//is ConditionTrue
			if cond.Type == v1beta1.ClusterReady && cond.Status != v1.ConditionTrue {
				glog.V(4).Infof("Ignoring cluster %v with %v condition status %v", cluster.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// clusterSyncLoop observes running clusters changes, and apply all services to new added cluster
// and add dns records for the changes
func (s *ServiceController) clusterSyncLoop() {
	var servicesToUpdate []*cachedService
	// should we remove cache for cluster from ready to not ready? should remove the condition predicate if no
	clusters, err := s.clusterStore.ClusterCondition(getClusterConditionPredicate()).List()
	if err != nil {
		glog.Infof("Fail to get cluster list")
		return
	}
	newClusters := clustersFromList(&clusters)
	var newSet, increase sets.String
	newSet = sets.NewString(newClusters...)
	if newSet.Equal(s.knownClusterSet) {
		// The set of cluster names in the services in the federation hasn't changed, but we can retry
		// updating any services that we failed to update last time around.
		servicesToUpdate = s.updateDNSRecords(servicesToUpdate, newClusters)
		return
	}
	glog.Infof("Detected change in list of cluster names. New  set: %v, Old set: %v", newSet, s.knownClusterSet)
	increase = newSet.Difference(s.knownClusterSet)
	// do nothing when cluster is removed.
	if increase != nil {
		// Try updating all services, and save the ones that fail to try again next
		// round.
		servicesToUpdate = s.serviceCache.allServices()
		numServices := len(servicesToUpdate)
		for newCluster := range increase {
			glog.Infof("New cluster observed %s", newCluster)
			s.updateAllServicesToCluster(servicesToUpdate, newCluster)
		}
		servicesToUpdate = s.updateDNSRecords(servicesToUpdate, newClusters)
		glog.Infof("Successfully updated %d out of %d DNS records to direct traffic to the updated cluster",
			numServices-len(servicesToUpdate), numServices)
	}
	s.knownClusterSet = newSet
}

func (s *ServiceController) updateAllServicesToCluster(services []*cachedService, clusterName string) {
	cluster, ok := s.clusterCache.clientMap[clusterName]
	if ok {
		for _, cachedService := range services {
			appliedState := cachedService.lastState
			s.processServiceForCluster(cachedService, clusterName, appliedState, cluster.clientset)
		}
	}
}

// updateDNSRecords updates all existing federation service DNS Records so that
// they will match the list of cluster names provided.
// Returns the list of services that couldn't be updated.
func (s *ServiceController) updateDNSRecords(services []*cachedService, clusters []string) (servicesToRetry []*cachedService) {
	for _, service := range services {
		func() {
			service.rwlock.Lock()
			defer service.rwlock.Unlock()
			// If the applied state is nil, that means it hasn't yet been successfully dealt
			// with by the DNS Record reconciler. We can trust the DNS Record
			// reconciler to ensure the federation service's DNS records are created to target
			// the correct backend service IP's
			if service.appliedState == nil {
				return
			}
			if err := s.lockedUpdateDNSRecords(service, clusters); err != nil {
				glog.Errorf("External error while updating DNS Records: %v.", err)
				servicesToRetry = append(servicesToRetry, service)
			}
		}()
	}
	return servicesToRetry
}

// lockedUpdateDNSRecords Updates the DNS records of a service, assuming we hold the mutex
// associated with the service.
func (s *ServiceController) lockedUpdateDNSRecords(service *cachedService, clusterNames []string) error {
	if !wantsDNSRecords(service.appliedState) {
		return nil
	}

	ensuredCount := 0
	unensuredCount := 0
	for key := range s.clusterCache.clientMap {
		for _, clusterName := range clusterNames {
			if key == clusterName {
				err := s.ensureDnsRecords(clusterName, service)
				if err != nil {
					unensuredCount += 1
					glog.V(4).Infof("Failed to update DNS records for service %v from cluster %s: %v", service, clusterName, err)
				} else {
					ensuredCount += 1
				}
			}
		}
	}
	missedCount := len(clusterNames) - ensuredCount - unensuredCount
	if missedCount > 0 || unensuredCount > 0 {
		return fmt.Errorf("Failed to update DNS records for %d clusters for service %v due to missing clients [missed count: %d] and/or failing to ensure DNS records [unensured count: %d]",
			len(clusterNames), service, missedCount, unensuredCount)
	}
	return nil
}

func LoadBalancerIPsAreEqual(oldService, newService *v1.Service) bool {
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

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedService) resetRetryDelay() {
	s.lastRetryDelay = time.Duration(0)
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedService) nextFedUpdateDelay() time.Duration {
	s.lastFedUpdateDelay = s.lastFedUpdateDelay * 2
	if s.lastFedUpdateDelay < minRetryDelay {
		s.lastFedUpdateDelay = minRetryDelay
	}
	if s.lastFedUpdateDelay > maxRetryDelay {
		s.lastFedUpdateDelay = maxRetryDelay
	}
	return s.lastFedUpdateDelay
}

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedService) resetFedUpdateDelay() {
	s.lastFedUpdateDelay = time.Duration(0)
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedService) nextDNSUpdateDelay() time.Duration {
	s.lastDNSUpdateDelay = s.lastDNSUpdateDelay * 2
	if s.lastDNSUpdateDelay < minRetryDelay {
		s.lastDNSUpdateDelay = minRetryDelay
	}
	if s.lastDNSUpdateDelay > maxRetryDelay {
		s.lastDNSUpdateDelay = maxRetryDelay
	}
	return s.lastDNSUpdateDelay
}

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedService) resetDNSUpdateDelay() {
	s.lastDNSUpdateDelay = time.Duration(0)
}

// syncService will sync the Service with the given key if it has had its expectations fulfilled,
// meaning it did not expect to see any more of its pods created or deleted. This function is not meant to be
// invoked concurrently with the same key.
func (s *ServiceController) syncService(key string) error {
	startTime := time.Now()
	var cachedService *cachedService
	var retryDelay time.Duration
	defer func() {
		glog.V(4).Infof("Finished syncing service %q (%v)", key, time.Now().Sub(startTime))
	}()
	// obj holds the latest service info from apiserver
	objFromStore, exists, err := s.serviceStore.Indexer.GetByKey(key)
	if err != nil {
		glog.Errorf("Unable to retrieve service %v from store: %v", key, err)
		s.queue.Add(key)
		return err
	}
	if !exists {
		// service absence in store means watcher caught the deletion, ensure LB info is cleaned
		glog.Infof("Service has been deleted %v", key)
		err, retryDelay = s.processServiceDeletion(key)
	}
	// Create a copy before modifying the obj to prevent race condition with
	// other readers of obj from store.
	obj, err := conversion.NewCloner().DeepCopy(objFromStore)
	if err != nil {
		glog.Errorf("Error in deep copying service %v retrieved from store: %v", key, err)
		s.queue.Add(key)
		return err
	}

	if exists {
		service, ok := obj.(*v1.Service)
		if ok {
			cachedService = s.serviceCache.getOrCreate(key)
			err, retryDelay = s.processServiceUpdate(cachedService, service, key)
		} else {
			tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
			if !ok {
				return fmt.Errorf("Object contained wasn't a service or a deleted key: %+v", obj)
			}
			glog.Infof("Found tombstone for %v", key)
			err, retryDelay = s.processServiceDeletion(tombstone.Key)
		}
	}

	if retryDelay != 0 {
		s.enqueueService(obj)
	} else if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to process service. Not retrying: %v", err))
	}
	return nil
}

// processServiceUpdate returns an error if processing the service update failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry in that Duration.
func (s *ServiceController) processServiceUpdate(cachedService *cachedService, service *v1.Service, key string) (error, time.Duration) {
	// Ensure that no other goroutine will interfere with our processing of the
	// service.
	cachedService.rwlock.Lock()
	defer cachedService.rwlock.Unlock()

	if service.DeletionTimestamp != nil {
		if err := s.delete(service); err != nil {
			glog.Errorf("Failed to delete %s: %v", service, err)
			s.eventRecorder.Eventf(service, api.EventTypeNormal, "DeleteFailed",
				"Service delete failed: %v", err)
			return err, cachedService.nextRetryDelay()
		}
		return nil, doNotRetry
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for service: %s",
		service.Name)
	// Add the required finalizers before creating a service in underlying clusters.
	updatedServiceObj, err := s.deletionHelper.EnsureFinalizers(service)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in service %s: %v",
			service.Name, err)
		return err, cachedService.nextRetryDelay()
	}
	service = updatedServiceObj.(*v1.Service)

	glog.V(3).Infof("Syncing service %s in underlying clusters", service.Name)

	// Update the cached service (used above for populating synthetic deletes)
	// alway trust service, which is retrieve from serviceStore, which keeps the latest service info getting from apiserver
	// if the same service is changed before this go routine finished, there will be another queue entry to handle that.
	cachedService.lastState = service
	err, retry := s.updateFederationService(key, cachedService)
	if err != nil {
		message := "Error occurs when updating service to all clusters"
		if retry {
			message += " (will retry): "
		} else {
			message += " (will not retry): "
		}
		message += err.Error()
		s.eventRecorder.Event(service, v1.EventTypeWarning, "UpdateServiceFail", message)
		return err, cachedService.nextRetryDelay()
	}
	// Always update the cache upon success.
	// NOTE: Since we update the cached service if and only if we successfully
	// processed it, a cached service being nil implies that it hasn't yet
	// been successfully processed.

	cachedService.appliedState = service
	s.serviceCache.set(key, cachedService)
	glog.V(4).Infof("Successfully proceeded services %s", key)
	cachedService.resetRetryDelay()
	return nil, doNotRetry
}

// delete deletes the given service or returns error if the deletion was not complete.
func (s *ServiceController) delete(service *v1.Service) error {
	glog.V(3).Infof("Handling deletion of service: %v", *service)
	_, err := s.deletionHelper.HandleObjectInUnderlyingClusters(service)
	if err != nil {
		return err
	}

	err = s.federationClient.Core().Services(service.Namespace).Delete(service.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of service finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the service and we do not have to do anything.
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete service: %v", err)
		}
	}
	return nil
}

// processServiceDeletion returns an error if processing the service deletion failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry in that Duration.
func (s *ServiceController) processServiceDeletion(key string) (error, time.Duration) {
	glog.V(2).Infof("Process service deletion for %v", key)
	s.serviceCache.delete(key)
	return nil, doNotRetry
}

func (s *ServiceController) deliverServiceObj(obj interface{}, delay time.Duration, failed bool) {
	object, ok := obj.(*v1.Service)
	if ok {
		s.deliverObject(types.NamespacedName{Namespace: object.Namespace, Name: object.Name}, delay, failed)
	} else {
		glog.Warningf("Unknown object recieved: %v", obj)
	}
}

func (s *ServiceController) deliverEndpointObj(obj interface{}, delay time.Duration, failed bool) {
	object, ok := obj.(*v1.Endpoints)
	if ok {
		s.deliverObject(types.NamespacedName{Namespace: object.Namespace, Name: object.Name}, delay, failed)
	} else {
		glog.Warningf("Unknown object recieved: %v", obj)
	}
}

func (s *ServiceController) deliverServicesOnClusterChange() {
	if !s.isSynced() {
		s.clusterDeliverer.DeliverAfter(allClustersKey, nil, s.clusterAvailableDelay)
	}
	glog.V(5).Infof("Delivering all service as cluster status changed")
	for _, obj := range s.serviceStore.Indexer.List() {
		service, ok := obj.(*v1.Service)
		if ok {
			s.deliverObject(types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, s.smallDelay, false)
		} else {
			glog.Warningf("Unknown object recieved: %v", obj)
		}
	}
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (s *ServiceController) deliverObject(object types.NamespacedName, delay time.Duration, failed bool) {
	key := object.String()
	if failed {
		s.flowcontrolBackoff.Next(key, time.Now())
		delay = delay + s.flowcontrolBackoff.Get(key)
	} else {
		s.flowcontrolBackoff.Reset(key)
	}
	s.objectDeliverer.DeliverAfter(key, &object, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet synced with
// the corresponding api server.
func (s *ServiceController) isSynced() bool {
	if !s.federatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	serviceClusters, err := s.federatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !s.federatedInformer.GetTargetStore().ClustersSynced(serviceClusters) {
		return false
	}

	if !s.endpointFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	endpointClusters, err := s.endpointFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !s.endpointFederatedInformer.GetTargetStore().ClustersSynced(endpointClusters) {
		return false
	}

	return true
}

// reconcileService triggers reconciliation of single federated service.
func (s *ServiceController) reconcileService(serviceKey types.NamespacedName) (reconciliationStatus, error) {
	if !s.isSynced() {
		glog.V(4).Infof("Data store not synced, delaying reconcilation: %v", serviceKey)
		return statusNotSynced, nil
	}

	key := serviceKey.String()
	fedServiceObjFromStore, exist, err := s.serviceStore.Indexer.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query federation service store for %s: %v", key, err)
		return statusError, err
	}

	if !exist {
		// Not federated service, ignoring.
		return statusAllOk, nil
	}

	glog.V(3).Infof("Reconciling federated service: %s", key)

	// Create a copy before modifying the service to prevent race condition with other readers of service from store
	fedServiceObj, err := api.Scheme.DeepCopy(fedServiceObjFromStore)
	fedService, ok := fedServiceObj.(*v1.Service)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %s, %v", key, err)
		return statusError, err
	}

	if fedService.DeletionTimestamp != nil {
		if err := s.delete(fedService); err != nil {
			glog.Errorf("Failed to delete %s: %v", key, err)
			s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteFailed", "Deleting service failed: %v", err)
			return statusError, err
		}
		glog.V(3).Infof("Deleting federated service succeeded: %s", key)
		s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteSucceed", "Deleting service succeeded")
		return statusAllOk, nil
	}

	// Add the required finalizers before creating a service in underlying clusters. This ensures that the
	// dependent services are deleted in underlying clusters when the federated service is deleted.
	updatedServiceObj, err := s.deletionHelper.EnsureFinalizers(fedService)
	if err != nil {
		glog.Warningf("Failed to ensure delete object from underlying clusters finalizer in service %s: %v", key, err)
		return statusError, err
	}
	fedService = updatedServiceObj.(*v1.Service)

	// Sync the service in all underlying ready clusters.
	clusters, err := s.federatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready cluster list: %v", err)
		return statusError, err
	}

	newLBIngress := si.NewLoadbalancerIngress()
	newServiceIngress := si.NewServiceIngress()
	operations := make([]fedutil.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterServiceObj, serviceFound, err := s.federatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s service from %s: %v", key, cluster.Name, err)
			return statusError, err
		}
		// Get Endpoints object corresponding to service from cluster informer store
		clusterEndpointsObj, endpointsFound, err :=
			s.endpointFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s endpoint from %s: %v", key, cluster.Name, err)
			return statusError, err
		}
		if !serviceFound {
			s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "CreateInCluster",
				"Creating service in cluster %s", cluster.Name)

			desiredService := &v1.Service{
				ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(fedService.ObjectMeta),
				Spec:       *(fedutil.DeepCopyApiTypeOrPanic(&fedService.Spec).(*v1.ServiceSpec)),
			}
			glog.V(4).Infof("Creating service in underlying cluster %s: %+v", cluster.Name, desiredService)

			desiredService.ResourceVersion = ""
			operations = append(operations, fedutil.FederatedOperation{
				Type:        fedutil.OperationTypeAdd,
				Obj:         desiredService,
				ClusterName: cluster.Name,
			})
		} else {
			clusterService := clusterServiceObj.(*v1.Service)

			desiredService := &v1.Service{
				ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(clusterService.ObjectMeta),
				Spec:       *(fedutil.DeepCopyApiTypeOrPanic(&fedService.Spec).(*v1.ServiceSpec)),
			}
			desiredService.Spec.ClusterIP = clusterService.Spec.ClusterIP
			if fedService.Spec.Type == v1.ServiceTypeLoadBalancer {
				for _, cPort := range clusterService.Spec.Ports {
					for i, fPort := range clusterService.Spec.Ports {
						if fPort.Name == cPort.Name && fPort.Protocol == cPort.Protocol &&
							fPort.Port == cPort.Port {
							desiredService.Spec.Ports[i].NodePort = cPort.NodePort
						}
					}
				}
			}

			// Update existing service, if needed.
			if !Equivalent(desiredService, clusterService) {
				glog.V(4).Infof("Service in underlying cluster %s does not match, Desired: %+v, "+
					"Existing: %+v", cluster.Name, desiredService, clusterService)
				s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInCluster",
					"Updating service in cluster %s. Desired: %+v\n Actual: %+v\n",
					cluster.Name, desiredService, clusterService)

				desiredService.ResourceVersion = clusterService.ResourceVersion
				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeUpdate,
					Obj:         desiredService,
					ClusterName: cluster.Name,
				})
			} else {
				glog.V(5).Infof("Service in underlying cluster %s match desired service: %+v", cluster.Name, desiredService)
			}

			// Get LoadBalancer status to update Federation Service
			addresses := []string{}
			if fedService.Spec.Type == v1.ServiceTypeLoadBalancer &&
				len(clusterService.Status.LoadBalancer.Ingress) > 0 {
				newLBIngress = append(newLBIngress, clusterService.Status.LoadBalancer.Ingress...)
				for _, ingress := range clusterService.Status.LoadBalancer.Ingress {
					if ingress.IP != "" {
						addresses = append(addresses, ingress.IP)
					} else if ingress.Hostname != "" {
						addresses = append(addresses, ingress.Hostname)
					}
				}
			}

			if fedService.Spec.Type == v1.ServiceTypeLoadBalancer {
				for _, address := range addresses {
					region := cluster.Status.Region
					// zone level - TODO might need other zone names for multi-zone clusters
					zone := cluster.Status.Zones[0]

					clusterIngress := newServiceIngress.GetOrCreateEmptyClusterServiceIngresses(region, zone, cluster.Name)
					clusterIngress.Endpoints = append(clusterIngress.Endpoints, address)
					clusterIngress.Healthy = false
					// if endpoints not found, load balancer ingress is treated as unhealthy
					if endpointsFound {
						clusterEndpoints := clusterEndpointsObj.(*v1.Endpoints)
						for _, subset := range clusterEndpoints.Subsets {
							if len(subset.Addresses) > 0 {
								clusterIngress.Healthy = true
								break
							}
						}
					}
					newServiceIngress.Endpoints[region][zone][cluster.Name] = clusterIngress
				}
			}
		}
	}

	// Update the federation service if there is any update in clustered service either is loadbalancer status or
	// endpoints update
	if fedService.Spec.Type == v1.ServiceTypeLoadBalancer {
		// Sort the endpoints information in annotations so that we can compare
		for _, region := range newServiceIngress.Endpoints {
			for _, zone := range region {
				for _, clusterIngress := range zone {
					sort.Strings(clusterIngress.Endpoints)
				}
			}
		}

		existingServiceIngress, err := si.ParseFederationServiceIngresses(fedService)
		if err != nil {
			glog.Errorf("Failed to parse endpoint annotations for service %s: %v", key, err)
			return statusError, err
		}

		// TODO: We should have a reliable cluster health check(should consider quorum) to detect cluster is not reachable
		// and remove dns records for them. Until a reliable cluster health check is available, below code is a workaround
		// to not remove the existing dns records which were created before the cluster went offline.
		unreadyClusters, err := s.federatedInformer.GetUnreadyClusters()
		if err != nil {
			glog.Errorf("Failed to get unready cluster list: %v", err)
			return statusError, err
		}
		for _, cluster := range unreadyClusters {
			region := cluster.Status.Region
			zone := cluster.Status.Zones[0]
			if existingServiceIngress.Endpoints != nil &&
				existingServiceIngress.Endpoints[region] != nil &&
				existingServiceIngress.Endpoints[region][zone] != nil &&
				existingServiceIngress.Endpoints[region][zone][cluster.Name] != nil {

				existingClusterIngress := existingServiceIngress.Endpoints[region][zone][cluster.Name]
				newClusterIngress := newServiceIngress.GetOrCreateEmptyClusterServiceIngresses(region, zone, cluster.Name)

				newClusterIngress.Endpoints = append(newClusterIngress.Endpoints, existingClusterIngress.Endpoints...)
				newClusterIngress.Healthy = existingClusterIngress.Healthy
				newServiceIngress.Endpoints[region][zone][cluster.Name] = newClusterIngress
				glog.V(5).Infof("Cluster %s is Offline, Preserving previously available status for Service %s/%s", cluster.Name, fedService.Namespace, fedService.Name)
			}
		}

		// Update federated service status and/or endpoints annotations if changed
		if !reflect.DeepEqual(existingServiceIngress, newServiceIngress) {
			var loadbalancerIngress []v1.LoadBalancerIngress

			if len(newLBIngress) > 0 {
				// Sort the endpoints so that we can compare
				sort.Sort(newLBIngress)
				loadbalancerIngress = make([]v1.LoadBalancerIngress, len(newLBIngress))
				for i, ingress := range newLBIngress {
					loadbalancerIngress[i] = v1.LoadBalancerIngress(ingress)
				}
				if !reflect.DeepEqual(fedService.Status.LoadBalancer.Ingress, loadbalancerIngress) {
					fedService.Status.LoadBalancer.Ingress = loadbalancerIngress
					glog.V(3).Infof("Federated service loadbalancer status updated for %s: %v", key, loadbalancerIngress)
				}
			}

			fedService = si.UpdateFederationServiceIngresses(fedService, newServiceIngress)
			glog.V(3).Infof("Federated service ingress endpoints health updated for %s: %v", key, newServiceIngress)
			_, err = s.federationClient.Core().Services(fedService.Namespace).UpdateStatus(fedService)
			if err != nil {
				glog.Errorf("Error updating the federation service object %s: %v", key, err)
				return statusError, err
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		glog.V(5).Infof("Everything is in order in underlying clusters")
		return statusAllOk, nil
	}
	glog.V(3).Infof("Adding/Updating service %s in underlying clusters. Operations: %d", key, len(operations))

	err = s.federatedUpdater.UpdateWithOnError(operations, s.updateTimeout,
		func(op fedutil.FederatedOperation, operror error) {
			glog.Errorf("Service update in cluster %s failed: %v", op.ClusterName, operror)
			s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInClusterFailed",
				"Service update in cluster %s failed: %v", op.ClusterName, operror)
		})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		if !apierrors.IsAlreadyExists(err) {
			return statusError, err
		}
	}

	// Some operations were made, reconcile after a while.
	return statusNeedRecheck, nil
}

// Equivalent Checks if cluster-independent, user provided data in two given services are equal. If in the future the
// services structure is expanded then any field that is not populated by the api server should be included here.
func Equivalent(s1, s2 *v1.Service) bool {
	if s1.Name != s2.Name {
		return false
	}
	if s1.Namespace != s2.Namespace {
		return false
	}
	if !reflect.DeepEqual(s1.Labels, s2.Labels) && (len(s1.Labels) != 0 || len(s2.Labels) != 0) {
		return false
	}
	if !reflect.DeepEqual(s1.Spec, s2.Spec) {
		return false
	}
	// TODO: should check for all annotations except FederationServiceIngressAnnotation

	return true
}
