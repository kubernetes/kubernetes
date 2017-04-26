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
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	cache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
	fedapi "k8s.io/kubernetes/federation/apis/federation"
	v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationcache "k8s.io/kubernetes/federation/client/cache"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	serviceSyncPeriod = 10 * time.Minute

	UserAgentName = "federation-service-controller"

	reviewDelay           = 10 * time.Second
	updateTimeout         = 30 * time.Second
	allClustersKey        = "ALL_CLUSTERS"
	clusterAvailableDelay = time.Second * 20
	ControllerName        = "services"
)

var (
	RequiredResources = []schema.GroupVersionResource{v1.SchemeGroupVersion.WithResource("services")}
)

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
	dnsZones dnsprovider.Zones
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
	queue *workqueue.Type

	// For triggering all services reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	deletionHelper *deletionhelper.DeletionHelper

	reviewDelay           time.Duration
	clusterAvailableDelay time.Duration
	updateTimeout         time.Duration

	endpointFederatedInformer fedutil.FederatedInformer
	federatedUpdater          fedutil.FederatedUpdater
	objectDeliverer           *util.DelayingDeliverer
	flowcontrolBackoff        *flowcontrol.Backoff
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
		dns:                   dns,
		federationClient:      federationClient,
		federationName:        federationName,
		serviceDnsSuffix:      serviceDnsSuffix,
		zoneName:              zoneName,
		zoneID:                zoneID,
		eventBroadcaster:      broadcaster,
		eventRecorder:         recorder,
		queue:                 workqueue.New(),
		reviewDelay:           reviewDelay,
		clusterAvailableDelay: clusterAvailableDelay,
		updateTimeout:         updateTimeout,
		flowcontrolBackoff:    flowcontrol.NewBackOff(5*time.Second, time.Minute),
	}
	s.objectDeliverer = util.NewDelayingDeliverer()
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
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) {
			glog.V(5).Infof("Delivering notification from federation: %v", obj)
			s.deliverObject(obj, 0, false)
		}),
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	s.serviceStore = corelisters.NewServiceLister(serviceIndexer)

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
					glog.V(5).Infof("Delivering service notification from federated cluster %s: %v", cluster.Name, obj)
					s.deliverObject(obj, s.reviewDelay, false)
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
			orphanDependents := false
			err := client.Core().Services(svc.Namespace).Delete(svc.Name, &metav1.DeleteOptions{OrphanDependents: &orphanDependents})
			// IsNotFound error is fine since that means the object is deleted already.
			if errors.IsNotFound(err) {
				return nil
			}
			return err
		})

	// Federated informers on endpoints in federated clusters.
	// This will enable to check if service ingress endpoints in federated clusters are reachable
	s.endpointFederatedInformer = fedutil.NewFederatedInformer(
		federationClient,
		func(cluster *v1beta1.Cluster, targetClient kubeclientset.Interface) (
			cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Core().Endpoints(metav1.NamespaceAll).List(options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Endpoints(metav1.NamespaceAll).Watch(options)
					},
				},
				&v1.Endpoints{},
				controller.NoResyncPeriodFunc(),
				fedutil.NewTriggerOnMetaAndFieldChanges(
					"Subsets",
					func(obj pkgruntime.Object) {
						glog.V(5).Infof("Delivering endpoint notification from federated cluster %s :%v", cluster.Name, obj)
						s.deliverObject(obj, s.reviewDelay, false)
					},
				))
		},
		&fedutil.ClusterLifecycleHandlerFuncs{},
	)

	s.deletionHelper = deletionhelper.NewDeletionHelper(
		s.updateService,
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

// Sends the given updated object to apiserver.
// Assumes that the given object is a service.
func (s *ServiceController) updateService(obj pkgruntime.Object) (pkgruntime.Object, error) {
	service := obj.(*v1.Service)
	return s.federationClient.Core().Services(service.Namespace).Update(service)
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
	s.endpointFederatedInformer.Start()
	s.objectDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		s.queue.Add(item.Value.(string))
	})
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		s.deliverServicesOnClusterChange()
	})
	fedutil.StartBackoffGC(s.flowcontrolBackoff, stopCh)
	go s.serviceController.Run(stopCh)

	for i := 0; i < workers; i++ {
		go wait.Until(s.fedServiceWorker, time.Second, stopCh)
	}
	go func() {
		<-stopCh
		glog.Infof("Shutting down Federation Service Controller")
		s.queue.ShutDown()
		s.federatedInformer.Stop()
		s.endpointFederatedInformer.Stop()
		s.objectDeliverer.Stop()
		s.clusterDeliverer.Stop()
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

type reconciliationStatus string

const (
	statusAllOk               = reconciliationStatus("ALL_OK")
	statusRecoverableError    = reconciliationStatus("RECOVERABLE_ERROR")
	statusNonRecoverableError = reconciliationStatus("NON_RECOVERABLE_ERROR")
	statusNotSynced           = reconciliationStatus("NOSYNC")
)

// fedServiceWorker runs a worker thread that just dequeues items, processes them, and marks them done.
func (s *ServiceController) fedServiceWorker() {
	for {
		func() {
			key, quit := s.queue.Get()
			if quit {
				return
			}
			defer s.queue.Done(key)
			service := key.(string)
			status := s.reconcileService(service)
			switch status {
			case statusAllOk:
				break
			case statusNotSynced:
				glog.V(5).Infof("Delivering notification for %q after clusterAvailableDelay", service)
				s.deliverService(service, s.clusterAvailableDelay, false)
			case statusRecoverableError:
				s.deliverService(service, 0, true)
			case statusNonRecoverableError:
				// error is already logged, do nothing
			default:
				// unreachable
			}
		}()
	}
}

func wantsDNSRecords(service *v1.Service) bool {
	return service.Spec.Type == v1.ServiceTypeLoadBalancer
}

// delete deletes the given service or returns error if the deletion was not complete.
func (s *ServiceController) delete(service *v1.Service) error {
	glog.V(3).Infof("Handling deletion of service: %v", *service)
	_, err := s.deletionHelper.HandleObjectInUnderlyingClusters(service)
	if err != nil {
		return err
	}

	// Ensure DNS records are removed for service
	if wantsDNSRecords(service) {
		key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
		serviceIngress, err := ParseFederatedServiceIngress(service)
		if err != nil {
			runtime.HandleError(fmt.Errorf("Failed to parse endpoint annotations for service %s: %v", key, err))
			return err
		}
		for _, ingress := range serviceIngress.Items {
			err := s.ensureDnsRecords(ingress.Cluster, service)
			if err != nil {
				glog.V(4).Infof("Error ensuring DNS Records for service %s on cluster %s: %v", key, ingress.Cluster, err)
				return err
			}
			glog.V(4).Infof("Ensured DNS records for Service %s in cluster %q", key, ingress.Cluster)
		}
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

func (s *ServiceController) deliverServicesOnClusterChange() {
	if !s.isSynced() {
		s.clusterDeliverer.DeliverAfter(allClustersKey, nil, s.clusterAvailableDelay)
	}
	glog.V(5).Infof("Delivering all service as cluster status changed")
	serviceList, err := s.serviceStore.List(labels.Everything())
	if err != nil {
		runtime.HandleError(fmt.Errorf("error listing federated services: %v", err))
		s.clusterDeliverer.DeliverAfter(allClustersKey, nil, 0)
	}
	for _, service := range serviceList {
		s.deliverObject(service, 0, false)
	}
}

func (s *ServiceController) deliverObject(object interface{}, delay time.Duration, failed bool) {
	switch value := object.(type) {
	case *v1.Service:
		s.deliverService(types.NamespacedName{Namespace: value.Namespace, Name: value.Name}.String(), delay, failed)
	case *v1.Endpoints:
		s.deliverService(types.NamespacedName{Namespace: value.Namespace, Name: value.Name}.String(), delay, failed)
	default:
		glog.Warningf("Unknown object received: %v", object)
	}
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (s *ServiceController) deliverService(key string, delay time.Duration, failed bool) {
	if failed {
		s.flowcontrolBackoff.Next(key, time.Now())
		delay = delay + s.flowcontrolBackoff.Get(key)
	} else {
		s.flowcontrolBackoff.Reset(key)
	}
	s.objectDeliverer.DeliverAfter(key, key, delay)
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
		runtime.HandleError(fmt.Errorf("Failed to get ready clusters: %v", err))
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
		runtime.HandleError(fmt.Errorf("Failed to get ready clusters: %v", err))
		return false
	}
	if !s.endpointFederatedInformer.GetTargetStore().ClustersSynced(endpointClusters) {
		return false
	}

	return true
}

// reconcileService triggers reconciliation of a federated service with corresponding services in federated clusters.
// This function is called on service Addition/Deletion/Updation either in federated cluster or in federation.
func (s *ServiceController) reconcileService(key string) reconciliationStatus {
	if !s.isSynced() {
		glog.V(4).Infof("Data store not synced, delaying reconcilation: %v", key)
		return statusNotSynced
	}

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Invalid key %q recieved, unable to split key to namespace and name, err: %v", key, err))
		return statusNonRecoverableError
	}

	service, err := s.serviceStore.Services(namespace).Get(name)
	if errors.IsNotFound(err) {
		// Not a federated service, ignoring.
		return statusAllOk
	} else if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to retrieve federated service %q from store: %v", key, err))
		return statusRecoverableError
	}

	glog.V(3).Infof("Reconciling federated service: %s", key)

	// Create a copy before modifying the service to prevent race condition with other readers of service from store
	fedServiceObj, err := api.Scheme.DeepCopy(service)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Error in copying obj: %s, %v", key, err))
		return statusNonRecoverableError
	}
	fedService, ok := fedServiceObj.(*v1.Service)
	if err != nil || !ok {
		runtime.HandleError(fmt.Errorf("Unknown obj recieved from store: %#v, %v", fedServiceObj, err))
		return statusNonRecoverableError
	}

	// Handle deletion of federated service
	if fedService.DeletionTimestamp != nil {
		if err := s.delete(fedService); err != nil {
			runtime.HandleError(fmt.Errorf("Failed to delete %s: %v", key, err))
			s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteFailed", "Deleting service failed: %v", err)
			return statusRecoverableError
		}
		glog.V(3).Infof("Deleting federated service succeeded: %s", key)
		s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteSucceed", "Deleting service succeeded")
		return statusAllOk
	}

	// Add the required finalizers before creating a service in underlying clusters. This ensures that the
	// dependent services in underlying clusters are deleted when the federated service is deleted.
	updatedServiceObj, err := s.deletionHelper.EnsureFinalizers(fedService)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to ensure setting finalizer for service %s: %v", key, err))
		return statusRecoverableError
	}
	fedService = updatedServiceObj.(*v1.Service)

	// Synchronize the federated service in all underlying ready clusters.
	clusters, err := s.federatedInformer.GetReadyClusters()
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get ready cluster list: %v", err))
		return statusRecoverableError
	}

	newLBStatus := newLoadbalancerStatus()
	newServiceIngress := NewFederatedServiceIngress()
	operations := make([]fedutil.FederatedOperation, 0)
	for _, cluster := range clusters {
		// Aggregate all operations to perform on all federated clusters
		operation, err := s.getOperationsToPerformOnCluster(cluster, fedService)
		if err != nil {
			return statusRecoverableError
		}
		if operation != nil {
			operations = append(operations, *operation)
		}

		// Aggregate LoadBalancerStatus from all services in federated clusters to update status in federated service
		lbStatus, err := s.getServiceStatusInCluster(cluster, key)
		if err != nil {
			return statusRecoverableError
		}
		if len(lbStatus.Ingress) > 0 {
			newLBStatus.Ingress = append(newLBStatus.Ingress, lbStatus.Ingress...)

			// Add/Update federated service ingress only if there are reachable endpoints backing the lb service
			endpoints, err := s.getServiceEndpointsInCluster(cluster, key)
			if err != nil {
				return statusRecoverableError
			}
			// if there are no endpoints created for the service then the loadbalancer ingress
			// is not reachable, so do not consider such loadbalancer ingresses for federated
			// service ingresses
			if len(endpoints) > 0 {
				clusterIngress := fedapi.ClusterServiceIngress{
					Cluster: cluster.Name,
					Items:   lbStatus.Ingress,
				}
				newServiceIngress.Items = append(newServiceIngress.Items, clusterIngress)
			}
		}
	}

	if len(operations) != 0 {
		err = s.federatedUpdater.UpdateWithOnError(operations, s.updateTimeout,
			func(op fedutil.FederatedOperation, operror error) {
				runtime.HandleError(fmt.Errorf("Service update in cluster %s failed: %v", op.ClusterName, operror))
				s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInClusterFailed", "Service update in cluster %s failed: %v", op.ClusterName, operror)
			})
		if err != nil {
			if !errors.IsAlreadyExists(err) {
				runtime.HandleError(fmt.Errorf("Failed to execute updates for %s: %v", key, err))
				return statusRecoverableError
			}
		}
	}

	// Update the federated service if there are any updates in clustered service (status/endpoints)
	err = s.updateFederatedService(fedService, newLBStatus, newServiceIngress)
	if err != nil {
		return statusRecoverableError
	}

	glog.V(5).Infof("Everything is in order in federated clusters for service %s", key)
	return statusAllOk
}

// getOperationsToPerformOnCluster returns the operations to be performed so that clustered service is in sync with federated service
func (s *ServiceController) getOperationsToPerformOnCluster(cluster *v1beta1.Cluster, fedService *v1.Service) (*fedutil.FederatedOperation, error) {
	var operation *fedutil.FederatedOperation

	key := types.NamespacedName{Namespace: fedService.Namespace, Name: fedService.Name}.String()
	clusterServiceObj, serviceFound, err := s.federatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get %s service from %s: %v", key, cluster.Name, err))
		return nil, err
	}
	if !serviceFound {
		desiredService := &v1.Service{
			ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(fedService.ObjectMeta),
			Spec:       *(fedutil.DeepCopyApiTypeOrPanic(&fedService.Spec).(*v1.ServiceSpec)),
		}
		desiredService.ResourceVersion = ""

		glog.V(4).Infof("Creating service in underlying cluster %s: %+v", cluster.Name, desiredService)
		s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "CreateInCluster", "Creating service in cluster %s", cluster.Name)

		operation = &fedutil.FederatedOperation{
			Type:        fedutil.OperationTypeAdd,
			Obj:         desiredService,
			ClusterName: cluster.Name,
		}
	} else {
		clusterService, ok := clusterServiceObj.(*v1.Service)
		if !ok {
			runtime.HandleError(fmt.Errorf("Unexpected error for %q: %v", key, err))
			return nil, err
		}

		desiredService := &v1.Service{
			ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(clusterService.ObjectMeta),
			Spec:       *(fedutil.DeepCopyApiTypeOrPanic(&fedService.Spec).(*v1.ServiceSpec)),
		}

		// ClusterIP and NodePort are allocated to Service by cluster, so retain the same if any while updating
		desiredService.Spec.ClusterIP = clusterService.Spec.ClusterIP
		for _, cPort := range clusterService.Spec.Ports {
			for i, fPort := range clusterService.Spec.Ports {
				if fPort.Name == cPort.Name && fPort.Protocol == cPort.Protocol && fPort.Port == cPort.Port {
					desiredService.Spec.Ports[i].NodePort = cPort.NodePort
				}
			}
		}

		// Update existing service, if needed.
		if !Equivalent(desiredService, clusterService) {
			glog.V(4).Infof("Service in underlying cluster %s does not match, Desired: %+v, Existing: %+v", cluster.Name, desiredService, clusterService)
			s.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInCluster", "Updating service in cluster %s. Desired: %+v\n Actual: %+v\n", cluster.Name, desiredService, clusterService)

			// ResourceVersion of cluster service can be different from federated service,
			// so do not update ResourceVersion while updating cluster service
			desiredService.ResourceVersion = clusterService.ResourceVersion

			operation = &fedutil.FederatedOperation{
				Type:        fedutil.OperationTypeUpdate,
				Obj:         desiredService,
				ClusterName: cluster.Name,
			}
		} else {
			glog.V(5).Infof("Service in underlying cluster %s is up to date: %+v", cluster.Name, desiredService)
		}
	}
	return operation, nil
}

// getServiceStatusInCluster returns service status in federated cluster
func (s *ServiceController) getServiceStatusInCluster(cluster *v1beta1.Cluster, key string) (*v1.LoadBalancerStatus, error) {
	lbStatus := &v1.LoadBalancerStatus{}

	clusterServiceObj, serviceFound, err := s.federatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get %s service from %s: %v", key, cluster.Name, err))
		return lbStatus, err
	}
	if serviceFound {
		clusterService, ok := clusterServiceObj.(*v1.Service)
		if !ok {
			err = fmt.Errorf("Unknown object received: %v", clusterServiceObj)
			runtime.HandleError(err)
			return lbStatus, err
		}
		lbStatus = &clusterService.Status.LoadBalancer
		newLbStatus := &loadbalancerStatus{*lbStatus}
		sort.Sort(newLbStatus)
	}
	return lbStatus, nil
}

// getServiceEndpointsInCluster returns ready endpoints corresonding to service in federated cluster
func (s *ServiceController) getServiceEndpointsInCluster(cluster *v1beta1.Cluster, key string) ([]v1.EndpointAddress, error) {
	addresses := []v1.EndpointAddress{}

	clusterEndpointsObj, endpointsFound, err := s.endpointFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get %s endpoint from %s: %v", key, cluster.Name, err))
		return addresses, err
	}
	if endpointsFound {
		clusterEndpoints, ok := clusterEndpointsObj.(*v1.Endpoints)
		if !ok {
			glog.Warningf("Unknown object received: %v", clusterEndpointsObj)
			return addresses, fmt.Errorf("Unknown object received: %v", clusterEndpointsObj)
		}
		for _, subset := range clusterEndpoints.Subsets {
			if len(subset.Addresses) > 0 {
				addresses = append(addresses, subset.Addresses...)
			}
		}
	}
	return addresses, nil
}

// updateFederatedService updates the federated service with aggregated lbStatus and serviceIngresses
// and also updates the dns records as needed
func (s *ServiceController) updateFederatedService(fedService *v1.Service, newLBStatus *loadbalancerStatus, newServiceIngress *FederatedServiceIngress) error {
	key := types.NamespacedName{Namespace: fedService.Namespace, Name: fedService.Name}.String()
	needUpdate := false

	// Sort the endpoints so that we can compare
	sort.Sort(newLBStatus)
	if !reflect.DeepEqual(fedService.Status.LoadBalancer.Ingress, newLBStatus.Ingress) {
		fedService.Status.LoadBalancer.Ingress = newLBStatus.Ingress
		glog.V(3).Infof("Federated service loadbalancer status updated for %s: %v", key, newLBStatus.Ingress)
		needUpdate = true
	}

	existingServiceIngress, err := ParseFederatedServiceIngress(fedService)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to parse endpoint annotations for service %s: %v", key, err))
		return err
	}

	// TODO: We should have a reliable cluster health check(should consider quorum) to detect cluster is not
	// reachable and remove dns records for them. Until a reliable cluster health check is available, below code is
	// a workaround to not remove the existing dns records which were created before the cluster went offline.
	unreadyClusters, err := s.federatedInformer.GetUnreadyClusters()
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get unready cluster list: %v", err))
		return err
	}
	for _, cluster := range unreadyClusters {
		lbIngress := existingServiceIngress.GetClusterLoadBalancerIngresses(cluster.Name)
		newServiceIngress.AddClusterLoadBalancerIngresses(cluster.Name, lbIngress)
		glog.V(5).Infof("Cluster %s is Offline, Preserving previously available status for Service %s", cluster.Name, key)
	}

	// Update federated service status and/or ingress annotations if changed
	sort.Sort(newServiceIngress)
	if !reflect.DeepEqual(existingServiceIngress.Items, newServiceIngress.Items) {
		fedService = UpdateIngressAnnotation(fedService, newServiceIngress)
		glog.V(3).Infof("Federated service loadbalancer ingress updated for %s: existing: %#v, desired: %#v", key, existingServiceIngress, newServiceIngress)
		needUpdate = true
	}

	if needUpdate {
		var err error
		fedService, err = s.federationClient.Core().Services(fedService.Namespace).UpdateStatus(fedService)
		if err != nil {
			runtime.HandleError(fmt.Errorf("Error updating the federation service object %s: %v", key, err))
			return err
		}
	}

	// Ensure DNS records based on Annotations in federated service for all federated clusters
	if needUpdate && wantsDNSRecords(fedService) {
		for _, ingress := range newServiceIngress.Items {
			err := s.ensureDnsRecords(ingress.Cluster, fedService)
			if err != nil {
				runtime.HandleError(fmt.Errorf("Error ensuring DNS Records for service %s on cluster %q: %v", key, ingress.Cluster, err))
				return err
			}
			glog.V(4).Infof("Ensured DNS records for Service %s in cluster %q", key, ingress.Cluster)
		}
	}
	return nil
}

// Equivalent Checks if cluster-independent, user provided data in two given services are equal. If in the future the
// services structure is expanded then any field that is not populated by the api server should be included here.
func Equivalent(s1, s2 *v1.Service) bool {
	// TODO: should also check for all annotations except FederationServiceIngressAnnotation
	return s1.Name == s2.Name && s1.Namespace == s2.Namespace &&
		(reflect.DeepEqual(s1.Labels, s2.Labels) || (len(s1.Labels) == 0 && len(s2.Labels) == 0)) &&
		reflect.DeepEqual(s1.Spec, s2.Spec)
}

type loadbalancerStatus struct {
	v1.LoadBalancerStatus
}

func newLoadbalancerStatus() *loadbalancerStatus {
	return &loadbalancerStatus{}
}

func (lbs loadbalancerStatus) Len() int {
	return len(lbs.Ingress)
}

func (lbs loadbalancerStatus) Less(i, j int) bool {
	ipComparison := strings.Compare(lbs.Ingress[i].IP, lbs.Ingress[j].IP)
	hostnameComparison := strings.Compare(lbs.Ingress[i].Hostname, lbs.Ingress[j].Hostname)
	if ipComparison < 0 || (ipComparison == 0 && hostnameComparison < 0) {
		return true
	}
	return false
}

func (lbs loadbalancerStatus) Swap(i, j int) {
	lbs.Ingress[i].IP, lbs.Ingress[j].IP = lbs.Ingress[j].IP, lbs.Ingress[i].IP
	lbs.Ingress[i].Hostname, lbs.Ingress[j].Hostname = lbs.Ingress[j].Hostname, lbs.Ingress[i].Hostname
}
