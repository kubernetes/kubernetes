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
