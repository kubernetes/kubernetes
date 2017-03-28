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
	"sync"
	"time"

	"reflect"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	cache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationcache "k8s.io/kubernetes/federation/client/cache"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	serviceSyncPeriod = 10 * time.Minute
	clusterSyncPeriod = 10 * time.Minute

	// How long to wait before retrying the processing of a service change.
	// If this changes, the sleep in hack/jenkins/e2e.sh before downing a cluster
	// should be changed appropriately.
	minRetryDelay = 5 * time.Second
	maxRetryDelay = 300 * time.Second

	// client retry count and interval is when accessing a remote kube-apiserver or federation apiserver
	// how many times should be attempted and how long it should sleep when failure occurs
	// the retry should be in short time so no exponential backoff
	clientRetryCount = 5

	retryable = true

	doNotRetry = time.Duration(0)

	UserAgentName = "federation-service-controller"
	KubeAPIQPS    = 20.0
	KubeAPIBurst  = 30

	maxNoOfClusters = 100

	updateTimeout         = 30 * time.Second
	allClustersKey        = "ALL_CLUSTERS"
	clusterAvailableDelay = time.Second * 20
	ControllerName        = "services"
)

var (
	RequiredResources = []schema.GroupVersionResource{v1.SchemeGroupVersion.WithResource("services")}
)

type cachedService struct {
	lastState *v1.Service
	// The state as successfully applied to the DNS server
	appliedState *v1.Service
	// cluster endpoint map hold subset info from kubernetes clusters
	// key clusterName
	// value is a flag that if there is ready address, 1 means there is ready address
	endpointMap map[string]int
	// serviceStatusMap map holds service status info from kubernetes clusters, keyed on clusterName
	serviceStatusMap map[string]v1.LoadBalancerStatus
	// Ensures only one goroutine can operate on this service at any given time.
	rwlock sync.Mutex
	// Controls error back-off for proceeding federation service to k8s clusters
	lastRetryDelay time.Duration
	// Controls error back-off for updating federation service back to federation apiserver
	lastFedUpdateDelay time.Duration
	// Controls error back-off for dns record update
	lastDNSUpdateDelay time.Duration
}

type serviceCache struct {
	rwlock sync.Mutex // protects serviceMap
	// federation service map contains all service received from federation apiserver
	// key serviceName
	fedServiceMap map[string]*cachedService
}

type ServiceController struct {
	dns              dnsprovider.Interface
	federationClient fedclientset.Interface
	federationName   string
	// serviceDnsSuffix is the DNS suffix we use when publishing service DNS names
	serviceDnsSuffix string
	// zoneName and zoneID are used to identify the zone in which to put records
	zoneName string
	zoneID   string
	// each federation should be configured with a single zone (e.g. "mycompany.com")
	dnsZones     dnsprovider.Zones
	serviceCache *serviceCache
	clusterCache *clusterClientCache
	// A store of services, populated by the serviceController
	serviceStore corelisters.ServiceLister
	// Watches changes to all services
	serviceController cache.Controller
	federatedInformer fedutil.FederatedInformer
	// A store of services, populated by the serviceController
	clusterStore federationcache.StoreToClusterLister
	// Watches changes to all services
	clusterController cache.Controller
	eventBroadcaster  record.EventBroadcaster
	eventRecorder     record.EventRecorder
	// services that need to be synced
	queue           *workqueue.Type
	knownClusterSet sets.String
	// endpoint worker map contains all the clusters registered with an indication that worker exist
	// key clusterName
	endpointWorkerMap map[string]bool
	// channel for worker to signal that it is going out of existence
	endpointWorkerDoneChan chan string
	// service worker map contains all the clusters registered with an indication that worker exist
	// key clusterName
	serviceWorkerMap map[string]bool
	// channel for worker to signal that it is going out of existence
	serviceWorkerDoneChan chan string

	// For triggering all services reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	deletionHelper *deletionhelper.DeletionHelper
}

// New returns a new service controller to keep DNS provider service resources
// (like Kubernetes Services and DNS server records for service discovery) in sync with the registry.

func New(federationClient fedclientset.Interface, dns dnsprovider.Interface,
	federationName, serviceDnsSuffix, zoneName string, zoneID string) *ServiceController {
	broadcaster := record.NewBroadcaster()
	// federationClient event is not supported yet
	// broadcaster.StartRecordingToSink(&unversioned_core.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: UserAgentName})

	s := &ServiceController{
		dns:              dns,
		federationClient: federationClient,
		federationName:   federationName,
		serviceDnsSuffix: serviceDnsSuffix,
		zoneName:         zoneName,
		zoneID:           zoneID,
		serviceCache:     &serviceCache{fedServiceMap: make(map[string]*cachedService)},
		clusterCache: &clusterClientCache{
			rwlock:    sync.Mutex{},
			clientMap: make(map[string]*clusterCache),
		},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
		queue:            workqueue.New(),
		knownClusterSet:  make(sets.String),
	}
	s.clusterDeliverer = util.NewDelayingDeliverer()
	var serviceIndexer cache.Indexer
	serviceIndexer, s.serviceController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return s.federationClient.Core().Services(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return s.federationClient.Core().Services(metav1.NamespaceAll).Watch(options)
			},
		},
		&v1.Service{},
		serviceSyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc: s.enqueueService,
			UpdateFunc: func(old, cur interface{}) {
				// there is case that old and new are equals but we still catch the event now.
				if !reflect.DeepEqual(old, cur) {
					s.enqueueService(cur)
				}
			},
			DeleteFunc: s.enqueueService,
		},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	s.serviceStore = corelisters.NewServiceLister(serviceIndexer)
	s.clusterStore.Store, s.clusterController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return s.federationClient.Federation().Clusters().List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return s.federationClient.Federation().Clusters().Watch(options)
			},
		},
		&v1beta1.Cluster{},
		clusterSyncPeriod,
		cache.ResourceEventHandlerFuncs{
			DeleteFunc: s.clusterCache.delFromClusterSet,
			AddFunc:    s.clusterCache.addToClientMap,
			UpdateFunc: func(old, cur interface{}) {
				oldCluster, ok := old.(*v1beta1.Cluster)
				if !ok {
					return
				}
				curCluster, ok := cur.(*v1beta1.Cluster)
				if !ok {
					return
				}
				if !reflect.DeepEqual(oldCluster.Spec, curCluster.Spec) {
					// update when spec is changed
					s.clusterCache.addToClientMap(cur)
				}

				pred := getClusterConditionPredicate()
				// only update when condition changed to ready from not-ready
				if !pred(*oldCluster) && pred(*curCluster) {
					s.clusterCache.addToClientMap(cur)
				}
				// did not handle ready -> not-ready
				// how could we stop a controller?
			},
		},
	)

	clusterLifecycle := fedutil.ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *v1beta1.Cluster) {
			s.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
		},
	}
	fedInformerFactory := func(cluster *v1beta1.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
					return targetClient.Core().Services(metav1.NamespaceAll).List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return targetClient.Core().Services(metav1.NamespaceAll).Watch(options)
				},
			},
			&v1.Service{},
			controller.NoResyncPeriodFunc(),
			// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
			// would be just confirmation that some service operation succeeded.
			util.NewTriggerOnAllChanges(
				func(obj pkgruntime.Object) {
					// TODO: Use this to enque services.
				},
			))
	}

	s.federatedInformer = fedutil.NewFederatedInformer(federationClient, fedInformerFactory, &clusterLifecycle)

	federatedUpdater := fedutil.NewFederatedUpdater(s.federatedInformer,
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
			err := client.Core().Services(svc.Namespace).Delete(svc.Name, &metav1.DeleteOptions{})
			return err
		})

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
		federatedUpdater,
	)

	s.endpointWorkerMap = make(map[string]bool)
	s.serviceWorkerMap = make(map[string]bool)
	s.endpointWorkerDoneChan = make(chan string, maxNoOfClusters)
	s.serviceWorkerDoneChan = make(chan string, maxNoOfClusters)
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

// Adds the given finalizers to the given objects ObjectMeta.
// Assumes that the given object is a service.
func (s *ServiceController) addFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	service := obj.(*v1.Service)
	service.ObjectMeta.Finalizers = append(service.ObjectMeta.Finalizers, finalizers...)
	service, err := s.federationClient.Core().Services(service.Namespace).Update(service)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to service %s: %v", finalizers, service.Name, err)
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

// Run starts a background goroutine that watches for changes to federation services
// and ensures that they have Kubernetes services created, updated or deleted appropriately.
// federationSyncPeriod controls how often we check the federation's services to
// ensure that the correct Kubernetes services (and associated DNS entries) exist.
// This is only necessary to fudge over failed watches.
// clusterSyncPeriod controls how often we check the federation's underlying clusters and
// their Kubernetes services to ensure that matching services created independently of the Federation
// (e.g. directly via the underlying cluster's API) are correctly accounted for.

// It's an error to call Run() more than once for a given ServiceController
// object.
func (s *ServiceController) Run(workers int, stopCh <-chan struct{}) error {
	if err := s.init(); err != nil {
		return err
	}
	defer runtime.HandleCrash()
	s.federatedInformer.Start()
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		// TODO: Use this new clusterDeliverer to reconcile services in new clusters.
	})
	go s.serviceController.Run(stopCh)
	go s.clusterController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(s.fedServiceWorker, time.Second, stopCh)
	}
	go wait.Until(s.clusterEndpointWorker, time.Second, stopCh)
	go wait.Until(s.clusterServiceWorker, time.Second, stopCh)
	go wait.Until(s.clusterSyncLoop, time.Second, stopCh)
	go func() {
		<-stopCh
		glog.Infof("Shutting down Federation Service Controller")
		s.queue.ShutDown()
		s.federatedInformer.Stop()
	}()
	return nil
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
		return fmt.Errorf("the dns provider does not support zone enumeration, which is required for creating dns records")
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

// fedServiceWorker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncService is never invoked concurrently with the same key.
func (s *ServiceController) fedServiceWorker() {
	for {
		func() {
			key, quit := s.queue.Get()
			if quit {
				return
			}

			defer s.queue.Done(key)
			err := s.syncService(key.(string))
			if err != nil {
				glog.Errorf("Error syncing service: %v", err)
			}
		}()
	}
}

func wantsDNSRecords(service *v1.Service) bool {
	return service.Spec.Type == v1.ServiceTypeLoadBalancer
}

// processServiceForCluster creates or updates service to all registered running clusters,
// update DNS records and update the service info with DNS entries to federation apiserver.
// the function returns any error caught
func (s *ServiceController) processServiceForCluster(cachedService *cachedService, clusterName string, service *v1.Service, client *kubeclientset.Clientset) error {
	if service.DeletionTimestamp != nil {
		glog.V(4).Infof("Service has already been marked for deletion %v", service.Name)
		return nil
	}
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
	hasErr := false
	var wg sync.WaitGroup
	for clusterName, cache := range s.clusterCache.clientMap {
		wg.Add(1)
		go func(cache *clusterCache, clusterName string) {
			defer wg.Done()
			err := s.processServiceForCluster(cachedService, clusterName, desiredService, cache.clientset)
			if err != nil {
				hasErr = true
			}
		}(cache, clusterName)
	}
	wg.Wait()
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

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		glog.Errorf("Unable to retrieve service %v from store: %v", key, err)
		s.queue.Add(key)
		return err
	}

	service, err := s.serviceStore.Services(namespace).Get(name)
	switch {
	case errors.IsNotFound(err):
		// service absence in store means watcher caught the deletion, ensure LB info is cleaned
		glog.Infof("Service has been deleted %v", key)
		err, retryDelay = s.processServiceDeletion(key)
	case err != nil:
		glog.Errorf("Unable to retrieve service %v from store: %v", key, err)
		s.queue.Add(key)
		return err
	default:
		// Create a copy before modifying the obj to prevent race condition with
		// other readers of obj from store.
		copy, err := conversion.NewCloner().DeepCopy(service)
		if err != nil {
			glog.Errorf("Error in deep copying service %v retrieved from store: %v", key, err)
			s.queue.Add(key)
			return err
		}
		service := copy.(*v1.Service)
		cachedService = s.serviceCache.getOrCreate(key)
		err, retryDelay = s.processServiceUpdate(cachedService, service, key)
	}

	if retryDelay != 0 {
		s.enqueueService(service)
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
		if !errors.IsNotFound(err) {
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
