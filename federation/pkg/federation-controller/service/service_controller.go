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
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/golang/glog"

	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	FederationServiceIngressAnnotation = "federation.kubernetes.io/service-ingress-endpoints"
	allServicesKey                     = "ALL_SERVICES"
)

var (
	reviewDelay           = 10 * time.Second
	clusterAvailableDelay = 20 * time.Second
	smallDelay            = 3 * time.Second
	updateTimeout         = 30 * time.Second
)

type ServiceController struct {
	// For triggering single service reconciliation or endpoint update. This is used when there is an
	// add/update/delete operation on a service in either federated API server or in some member of the federation.
	objectDeliverer *util.DelayingDeliverer

	// For triggering all services reconciliation. This is used when a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains services present in members of federation.
	serviceFederatedInformer util.FederatedInformer
	// Contains endpoints present in members of federation.
	endpointFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedServiceUpdater util.FederatedUpdater
	// Definitions of services that should be federated.
	serviceInformerStore cache.StoreToServiceLister
	// Informer controller for services that should be federated.
	serviceInformerController cache.ControllerInterface

	// Client to federated api server.
	federationClient fedclientset.Interface

	// ServiceBackoff manager
	flowcontrolBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	serviceWorkQueue workqueue.Interface

	deletionHelper *deletionhelper.DeletionHelper

	reviewDelay           time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewServiceController returns a new service controller
func NewServiceController(client fedclientset.Interface) *ServiceController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(apiv1.EventSource{Component: "federated-service-controller"})

	sc := &ServiceController{
		federationClient:      client,
		reviewDelay:           reviewDelay,
		clusterAvailableDelay: clusterAvailableDelay,
		smallDelay:            smallDelay,
		updateTimeout:         updateTimeout,
		serviceWorkQueue:      workqueue.New(),
		flowcontrolBackoff:    flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build deliverers for triggering reconciliations.
	sc.objectDeliverer = util.NewDelayingDeliverer()
	sc.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on services that should be federated.
	sc.serviceInformerStore.Indexer, sc.serviceInformerController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
				return client.Core().Services(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return client.Core().Services(apiv1.NamespaceAll).Watch(options)
			},
		},
		&apiv1.Service{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj runtime.Object) {
			glog.V(5).Infof("Delivering notification from federation: %v", obj)
			sc.deliverServiceObj(obj, 0, false)
		}),
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	// Federated informer on services in members of federation.
	sc.serviceFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *fedv1.Cluster, targetClient kubeclientset.Interface) (
			cache.Store, cache.ControllerInterface) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
						return targetClient.Core().Services(apiv1.NamespaceAll).List(options)
					},
					WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Services(apiv1.NamespaceAll).Watch(options)
					},
				},
				&apiv1.Service{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most
				// cases it would be just confirmation that some service operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj runtime.Object) {
						glog.V(5).Infof("Delivering service notification from federated "+
							"cluster %s: %v", cluster.Name, obj)
						sc.deliverServiceObj(obj, sc.reviewDelay, false)
					},
				))
		},
		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *fedv1.Cluster) {
				// When new cluster becomes available, process all the services again.
				sc.clusterDeliverer.DeliverAfter(allServicesKey, nil, sc.clusterAvailableDelay)
			},
			ClusterUnavailable: func(cluster *fedv1.Cluster, _ []interface{}) {
				// When cluster is unregistered, process all the services again.
				sc.clusterDeliverer.DeliverAfter(allServicesKey, nil, 0)
			},
		},
	)

	// Federated informer on endpoints in members of federation to know backing healthy endpoints
	sc.endpointFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *fedv1.Cluster, targetClient kubeclientset.Interface) (
			cache.Store, cache.ControllerInterface) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
						return targetClient.Core().Endpoints(apiv1.NamespaceAll).List(options)
					},
					WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().Endpoints(apiv1.NamespaceAll).Watch(options)
					},
				},
				&apiv1.Endpoints{},
				time.Minute*10, // rechecks all endpoints on interval
				util.NewTriggerOnMetaAndFieldChanges(
					"Subsets",
					func(obj runtime.Object) {
						glog.V(5).Infof("Delivering endpoint notification from federated "+
							"cluster %s :%v", cluster.Name, obj)
						sc.deliverEndpointObj(obj, sc.reviewDelay, false)
					},
				))
		},
		&util.ClusterLifecycleHandlerFuncs{},
	)

	// Federated updater along with Create/Update/Delete operations.
	sc.federatedServiceUpdater = util.NewFederatedUpdater(sc.serviceFederatedInformer,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			service := obj.(*apiv1.Service)
			_, err := client.Core().Services(service.Namespace).Create(service)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			service := obj.(*apiv1.Service)
			_, err := client.Core().Services(service.Namespace).Update(service)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			service := obj.(*apiv1.Service)
			err := client.Core().Services(service.Namespace).Delete(service.Name, &apiv1.DeleteOptions{})
			// IsNotFound error is fine since that means the object is deleted already.
			if errors.IsNotFound(err) {
				return nil
			}
			return err
		})

	sc.deletionHelper = deletionhelper.NewDeletionHelper(
		sc.hasFinalizerFunc,
		sc.removeFinalizerFunc,
		sc.addFinalizerFunc,
		// objNameFunc
		func(obj runtime.Object) string {
			service := obj.(*apiv1.Service)
			return service.Name
		},
		sc.updateTimeout,
		sc.eventRecorder,
		sc.serviceFederatedInformer,
		sc.federatedServiceUpdater,
	)
	return sc
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (sc *ServiceController) hasFinalizerFunc(obj runtime.Object, finalizer string) bool {
	service := obj.(*apiv1.Service)
	for i := range service.ObjectMeta.Finalizers {
		if string(service.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta. Assumes that the given object is a service.
func (sc *ServiceController) removeFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	service := obj.(*apiv1.Service)
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
	service, err := sc.federationClient.Core().Services(service.Namespace).Update(service)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from service %s: %v",
			finalizer, service.Name, err)
	}
	return service, nil
}

// Adds the given finalizer to the given objects ObjectMeta. Assumes that the given object is a service.
func (sc *ServiceController) addFinalizerFunc(obj runtime.Object, finalizer string) (runtime.Object, error) {
	service := obj.(*apiv1.Service)
	service.ObjectMeta.Finalizers = append(service.ObjectMeta.Finalizers, finalizer)
	service, err := sc.federationClient.Core().Services(service.Namespace).Update(service)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizer %s to service %s: %v", finalizer, service.Name, err)
	}
	return service, nil
}

// Run runs the informer controllers
func (sc *ServiceController) Run(workers int, stopChan <-chan struct{}) {
	sc.objectDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		sc.serviceWorkQueue.Add(item.Value.(*types.NamespacedName))
	})
	sc.clusterDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		sc.reconcileServicesOnClusterChange()
	})

	for i := 0; i < workers; i++ {
		go wait.Until(sc.worker, time.Second, stopChan)
	}

	util.StartBackoffGC(sc.flowcontrolBackoff, stopChan)

	go sc.serviceInformerController.Run(stopChan)
	sc.serviceFederatedInformer.Start()
	sc.endpointFederatedInformer.Start()
	go func() {
		<-stopChan
		sc.serviceFederatedInformer.Stop()
		sc.endpointFederatedInformer.Stop()
		sc.objectDeliverer.Stop()
		sc.clusterDeliverer.Stop()
		sc.serviceWorkQueue.ShutDown()
	}()
}

func (sc *ServiceController) deliverServiceObj(obj interface{}, delay time.Duration, failed bool) {
	object := obj.(*apiv1.Service)
	sc.deliverObject(types.NamespacedName{Namespace: object.Namespace, Name: object.Name}, delay, failed)
}

func (sc *ServiceController) deliverEndpointObj(obj interface{}, delay time.Duration, failed bool) {
	object := obj.(*apiv1.Endpoints)
	sc.deliverObject(types.NamespacedName{Namespace: object.Namespace, Name: object.Name}, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (sc *ServiceController) deliverObject(object types.NamespacedName, delay time.Duration, failed bool) {
	key := object.String()
	if failed {
		sc.flowcontrolBackoff.Next(key, time.Now())
		delay = delay + sc.flowcontrolBackoff.Get(key)
	} else {
		sc.flowcontrolBackoff.Reset(key)
	}
	sc.objectDeliverer.DeliverAfter(key, &object, delay)
}

type reconciliationStatus string

const (
	statusAllOk       = reconciliationStatus("ALL_OK")
	statusNeedRecheck = reconciliationStatus("RECHECK")
	statusError       = reconciliationStatus("ERROR")
	statusNotSynced   = reconciliationStatus("NOSYNC")
)

func (sc *ServiceController) worker() {
	for {
		item, quit := sc.serviceWorkQueue.Get()
		if quit {
			return
		}
		serviceName := *item.(*types.NamespacedName)
		status, err := sc.reconcileService(serviceName)
		sc.serviceWorkQueue.Done(item)
		if err != nil {
			glog.Errorf("Error reconciling service: %v, delivering again", err)
			sc.deliverObject(serviceName, 0, true)
		} else {
			switch status {
			case statusAllOk:
				break
			case statusError:
				glog.V(5).Infof("Delivering notification again upon error %v:", serviceName)
				sc.deliverObject(serviceName, 0, true)
			case statusNeedRecheck:
				glog.V(5).Infof("Delivering notification again for recheck %v:", serviceName)
				sc.deliverObject(serviceName, reviewDelay, false)
			case statusNotSynced:
				glog.V(5).Infof("Delivering notification after clusterAvailableDelay %v:", serviceName)
				sc.deliverObject(serviceName, clusterAvailableDelay, false)
			default:
				glog.Errorf("Unhandled reconciliation status: %s, delivering again", status)
				sc.deliverObject(serviceName, reviewDelay, false)
			}
		}
	}
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet synced with
// the corresponding api server.
func (sc *ServiceController) isSynced() bool {
	if !sc.serviceFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	serviceClusters, err := sc.serviceFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !sc.serviceFederatedInformer.GetTargetStore().ClustersSynced(serviceClusters) {
		return false
	}

	if !sc.endpointFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	endpointClusters, err := sc.endpointFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !sc.endpointFederatedInformer.GetTargetStore().ClustersSynced(endpointClusters) {
		return false
	}

	return true
}

// reconcileServicesOnClusterChange triggers reconciliation of all federated services.
func (sc *ServiceController) reconcileServicesOnClusterChange() {
	if !sc.isSynced() {
		sc.clusterDeliverer.DeliverAfter(allServicesKey, nil, sc.clusterAvailableDelay)
	}
	glog.V(5).Infof("Delivering all service as cluster status changed")
	for _, obj := range sc.serviceInformerStore.Indexer.List() {
		service := obj.(*apiv1.Service)
		sc.deliverObject(types.NamespacedName{Namespace: service.Namespace, Name: service.Name},
			sc.smallDelay, false)
	}
}

// reconcileService triggers reconciliation of single federated service.
func (sc *ServiceController) reconcileService(serviceKey types.NamespacedName) (reconciliationStatus, error) {
	if !sc.isSynced() {
		glog.V(4).Infof("Data store not synced, delaying reconcilation: %v", serviceKey)
		return statusNotSynced, nil
	}

	key := serviceKey.String()
	fedServiceObjFromStore, exist, err := sc.serviceInformerStore.Indexer.GetByKey(key)
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
	fedService, ok := fedServiceObj.(*apiv1.Service)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj from store: %s, %v", key, err)
		return statusError, err
	}

	if fedService.DeletionTimestamp != nil {
		if err := sc.deleteService(fedService); err != nil {
			glog.Errorf("Failed to delete %s: %v", key, err)
			sc.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteFailed",
				"Deleting service failed: %v", err)
			return statusError, err
		}
		glog.V(3).Infof("Deleting federated service succeeded: %s", key)
		sc.eventRecorder.Eventf(fedService, api.EventTypeNormal, "DeleteSucceed", "Deleting service succeeded")
		return statusAllOk, nil
	}

	// Add the required finalizers before creating a service in underlying clusters. This ensures that the
	// dependent services are deleted in underlying clusters when the federated service is deleted.
	updatedServiceObj, err := sc.deletionHelper.EnsureFinalizers(fedService)
	if err != nil {
		glog.Warningf("Failed to ensure delete object from underlying clusters finalizer in service %s: %v",
			key, err)
		return statusError, err
	}
	fedService = updatedServiceObj.(*apiv1.Service)

	// Sync the service in all underlying ready clusters.
	clusters, err := sc.serviceFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready cluster list: %v", err)
		return statusError, err
	}

	newLBIngress := ingress{}
	newServiceIngress := &fed.FederatedServiceIngress{}
	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterServiceObj, serviceFound, err :=
			sc.serviceFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s service from %s: %v", key, cluster.Name, err)
			return statusError, err
		}
		// Get Endpoints object corresponding to service from cluster informer store
		clusterEndpointsObj, endpointsFound, err :=
			sc.endpointFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s endpoint from %s: %v", key, cluster.Name, err)
			return statusError, err
		}
		if !serviceFound {
			sc.eventRecorder.Eventf(fedService, api.EventTypeNormal, "CreateInCluster",
				"Creating service in cluster %s", cluster.Name)

			desiredService := &apiv1.Service{
				ObjectMeta: util.DeepCopyRelevantObjectMeta(fedService.ObjectMeta),
				Spec:       *(util.DeepCopyApiTypeOrPanic(&fedService.Spec).(*apiv1.ServiceSpec)),
			}
			glog.V(4).Infof("Creating service in underlying cluster %s: %+v", cluster.Name, desiredService)

			desiredService.ResourceVersion = ""
			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredService,
				ClusterName: cluster.Name,
			})
		} else {
			clusterService := clusterServiceObj.(*apiv1.Service)

			desiredService := &apiv1.Service{
				ObjectMeta: util.DeepCopyRelevantObjectMeta(clusterService.ObjectMeta),
				Spec:       *(util.DeepCopyApiTypeOrPanic(&fedService.Spec).(*apiv1.ServiceSpec)),
			}
			desiredService.Spec.ClusterIP = clusterService.Spec.ClusterIP
			if fedService.Spec.Type == apiv1.ServiceTypeLoadBalancer {
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
				glog.V(4).Infof("Service in underlying cluster %s doe not match, Desired: %+v, "+
					"Existing: %+v", cluster.Name, desiredService, clusterService)
				sc.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInCluster",
					"Updating service in cluster %s. Desired: %+v\n Actual: %+v\n",
					cluster.Name, desiredService, clusterService)

				desiredService.ResourceVersion = clusterService.ResourceVersion
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredService,
					ClusterName: cluster.Name,
				})
			} else {
				glog.V(5).Infof("Service in underlying cluster %s match desired service: %+v",
					cluster.Name, desiredService)
			}

			addresses := []string{}
			// Get LoadBalancer status to update Federation Service
			if fedService.Spec.Type == apiv1.ServiceTypeLoadBalancer &&
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

			if fedService.Spec.Type == apiv1.ServiceTypeLoadBalancer {
				for _, address := range addresses {
					region := cluster.Status.Region
					// zone level - TODO might need other zone names for multi-zone clusters
					zone := cluster.Status.Zones[0]

					clusterIngress :=
						getOrCreateClusterServiceIngresses(newServiceIngress,
							region, zone, cluster.Name)
					clusterIngress.Endpoints = append(clusterIngress.Endpoints, address)
					clusterIngress.Healthy = false
					// if endpoints not found, load balancer ingress is treated as unhealthy
					if endpointsFound {
						clusterEndpoints := clusterEndpointsObj.(*apiv1.Endpoints)
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
	if fedService.Spec.Type == apiv1.ServiceTypeLoadBalancer {
		// Sort the endpoints information in annotations so that we can compare
		for _, region := range newServiceIngress.Endpoints {
			for _, zone := range region {
				for _, clusterIngress := range zone {
					sort.Strings(clusterIngress.Endpoints)
				}
			}
		}

		existingServiceIngress, err := ParseFederationServiceIngresses(fedService)
		if err != nil {
			glog.Errorf("Failed to parse endpoint annotations for service %s: %v", key, err)
			return statusError, err
		}

		// Copy ingresses of unreachable clusters for comparison and do not delete those ingresses
		unreadyClusters, err := sc.serviceFederatedInformer.GetUnreadyClusters()
		if err != nil {
			glog.Errorf("Failed to get unready cluster list: %v", err)
			return statusError, err
		}

		for _, cluster := range unreadyClusters {
			region := cluster.Status.Region
			zone := cluster.Status.Zones[0]
			clusterIngress := getClusterServiceIngresses(existingServiceIngress, region, zone, cluster.Name)
			if clusterIngress != nil {
				getOrCreateClusterServiceIngresses(newServiceIngress, region, zone, cluster.Name)
				newServiceIngress.Endpoints[region][zone][cluster.Name] = clusterIngress
			}
		}

		// Update federated service status and/or endpoints annotations if changed
		if !reflect.DeepEqual(existingServiceIngress, newServiceIngress) {
			var loadbalancerIngress []apiv1.LoadBalancerIngress

			if len(newLBIngress) > 0 {
				// Sort the endpoints so that we can compare
				sort.Sort(newLBIngress)
				loadbalancerIngress = make([]apiv1.LoadBalancerIngress, len(newLBIngress))
				for i, ingress := range newLBIngress {
					loadbalancerIngress[i] = apiv1.LoadBalancerIngress(ingress)
				}
				if !reflect.DeepEqual(fedService.Status.LoadBalancer.Ingress, loadbalancerIngress) {
					fedService.Status.LoadBalancer.Ingress = loadbalancerIngress
					glog.V(3).Infof("Federated service loadbalancer status updated for %s: %v",
						key, loadbalancerIngress)
				}
			}

			fedService = updateFederationServiceIngresses(fedService, newServiceIngress)
			glog.V(3).Infof("Federated service ingress endpoints health updated for %s: %v",
				key, newServiceIngress)
			_, err = sc.federationClient.Core().Services(fedService.Namespace).UpdateStatus(fedService)
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

	err = sc.federatedServiceUpdater.UpdateWithOnError(operations, sc.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			glog.Errorf("Service update in cluster %s failed: %v", op.ClusterName, operror)
			sc.eventRecorder.Eventf(fedService, api.EventTypeNormal, "UpdateInClusterFailed",
				"Service update in cluster %s failed: %v", op.ClusterName, operror)
		})
	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		if !errors.IsAlreadyExists(err) {
			return statusError, err
		}
	}

	// Some operations were made, reconcile after a while.
	return statusNeedRecheck, nil
}

// deleteService deletes the given service or returns error if the deletion was not complete.
func (sc *ServiceController) deleteService(service *apiv1.Service) error {
	glog.V(3).Infof("Handling deletion of service: %v", *service)
	_, err := sc.deletionHelper.HandleObjectInUnderlyingClusters(service)
	if err != nil {
		return err
	}

	err = sc.federationClient.Core().Services(service.Namespace).Delete(service.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to
		// do anything. This is expected when we are processing an update as a result of service finalizer
		// deletion. The process that deleted the last finalizer is also going to delete the service and we do
		// not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete service: %v", err)
		}
	}
	return nil
}

// Equivalent Checks if cluster-independent, user provided data in two given services are equal. If in the future the
// services structure is expanded then any field that is not populated by the api server should be included here.
func Equivalent(s1, s2 *apiv1.Service) bool {
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

type ingress []apiv1.LoadBalancerIngress

func (slice ingress) Len() int {
	return len(slice)
}

func (slice ingress) Less(i, j int) bool {
	ipComparison := strings.Compare(slice[i].IP, slice[j].IP)
	hostnameComparison := strings.Compare(slice[i].Hostname, slice[j].Hostname)
	if ipComparison < 0 || (ipComparison == 0 && hostnameComparison < 0) {
		return true
	}
	return false
}

func (slice ingress) Swap(i, j int) {
	slice[i].IP, slice[j].IP = slice[j].IP, slice[i].IP
	slice[i].Hostname, slice[j].Hostname = slice[j].Hostname, slice[i].Hostname
}

// getClusterServiceIngresses returns cluster service ingresses for given region, zone and cluster name
func getClusterServiceIngresses(ingress *fed.FederatedServiceIngress,
	region, zone, cluster string) *fed.ClusterServiceIngress {
	if ingress.Endpoints != nil && ingress.Endpoints[region] != nil && ingress.Endpoints[region][zone] != nil &&
		ingress.Endpoints[region][zone][cluster] != nil {
		return ingress.Endpoints[region][zone][cluster]
	}
	return nil
}

// getOrCreateClusterServiceIngresses returns cluster service ingresses for given region, zone and cluster name if
// exist otherwise creates one and returns
func getOrCreateClusterServiceIngresses(ingress *fed.FederatedServiceIngress,
	region, zone, cluster string) *fed.ClusterServiceIngress {
	if ingress.Endpoints == nil {
		ingress.Endpoints = make(map[string]map[string]map[string]*fed.ClusterServiceIngress)
	}
	if ingress.Endpoints[region] == nil {
		ingress.Endpoints[region] = make(map[string]map[string]*fed.ClusterServiceIngress)
	}
	if ingress.Endpoints[region][zone] == nil {
		ingress.Endpoints[region][zone] = make(map[string]*fed.ClusterServiceIngress)
	}
	if ingress.Endpoints[region][zone][cluster] == nil {
		return &fed.ClusterServiceIngress{}
	}

	return ingress.Endpoints[region][zone][cluster]
}

// updateFederationServiceIngresses updates the federation service with service ingress annotation
func updateFederationServiceIngresses(fs *apiv1.Service, ingress *fed.FederatedServiceIngress) *apiv1.Service {
	annotationBytes, _ := json.Marshal(ingress)
	annotations := string(annotationBytes[:])
	if fs.Annotations == nil {
		fs.Annotations = make(map[string]string)
	}
	fs.Annotations[FederationServiceIngressAnnotation] = annotations
	return fs
}

// ParseFederationServiceIngresses extracts federation service ingresses from federation service
func ParseFederationServiceIngresses(fs *apiv1.Service) (*fed.FederatedServiceIngress, error) {
	fsIngress := fed.FederatedServiceIngress{}
	if fs.Annotations == nil {
		return &fsIngress, nil
	}
	fsIngressString, found := fs.Annotations[FederationServiceIngressAnnotation]
	if !found {
		return &fsIngress, nil
	}
	if err := json.Unmarshal([]byte(fsIngressString), &fsIngress); err != nil {
		return &fsIngress, err
	}
	return &fsIngress, nil
}

type EndpointMap fed.FederatedServiceIngress

func NewEpMap() *EndpointMap {
	return &EndpointMap{}
}

func (epMap *EndpointMap) AddEndpoint(region, zone, cluster string, endpoints []string, healthy bool) *EndpointMap {
	if epMap.Endpoints == nil {
		epMap.Endpoints = make(map[string]map[string]map[string]*fed.ClusterServiceIngress)
	}
	if epMap.Endpoints[region] == nil {
		epMap.Endpoints[region] = make(map[string]map[string]*fed.ClusterServiceIngress)
	}
	if epMap.Endpoints[region][zone] == nil {
		epMap.Endpoints[region][zone] = make(map[string]*fed.ClusterServiceIngress)
	}
	clusterIngEps, ok := epMap.Endpoints[region][zone][cluster]
	if !ok {
		clusterIngEps = &fed.ClusterServiceIngress{}
	}
	clusterIngEps.Endpoints = append(clusterIngEps.Endpoints, endpoints...)
	clusterIngEps.Healthy = healthy
	epMap.Endpoints[region][zone][cluster] = clusterIngEps
	return epMap
}

func (epMap *EndpointMap) RemoveEndpoint(region, zone, cluster string, endpoint string) *EndpointMap {
	clusterIngress := epMap.Endpoints[region][zone][cluster]
	for i, ep := range clusterIngress.Endpoints {
		if ep == endpoint {
			clusterIngress.Endpoints = append(clusterIngress.Endpoints[:i], clusterIngress.Endpoints[i+1:]...)
		}
	}
	if len(clusterIngress.Endpoints) == 0 {
		clusterIngress.Healthy = false
	}
	epMap.Endpoints[region][zone][cluster] = clusterIngress
	return epMap
}

func (epMap *EndpointMap) GetJSONMarshalledBytes() []byte {
	byteArray, _ := json.Marshal(*epMap)
	return byteArray
}

func NewEndpointAnnotation(byteArray []byte) map[string]string {
	epAnnotation := make(map[string]string)
	epAnnotation[FederationServiceIngressAnnotation] = string(byteArray[:])
	return epAnnotation
}
