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

package sync

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/clusterselector"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
)

// FederationSyncController synchronizes the state of a federated type
// to clusters that are members of the federation.
type FederationSyncController struct {
	// For triggering reconciliation of a single resource. This is
	// used when there is an add/update/delete operation on a resource
	// in either federated API server or in some member of the
	// federation.
	deliverer *util.DelayingDeliverer

	// For triggering reconciliation of all target resources. This is
	// used when a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains resources present in members of federation.
	informer util.FederatedInformer
	// For updating members of federation.
	updater util.FederatedUpdater
	// Definitions of resources that should be federated.
	store cache.Store
	// Informer controller for resources that should be federated.
	controller cache.Controller

	// Work queue allowing parallel processing of resources
	workQueue workqueue.Interface

	// Backoff manager
	backoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	reviewDelay             time.Duration
	clusterAvailableDelay   time.Duration
	clusterUnavailableDelay time.Duration
	smallDelay              time.Duration
	updateTimeout           time.Duration

	adapter federatedtypes.FederatedTypeAdapter
}

// StartFederationSyncController starts a new sync controller for a type adapter
func StartFederationSyncController(kind string, adapterFactory federatedtypes.AdapterFactory, config *restclient.Config, stopChan <-chan struct{}, minimizeLatency bool, adapterSpecificArgs map[string]interface{}) {
	restclient.AddUserAgent(config, fmt.Sprintf("federation-%s-controller", kind))
	client := federationclientset.NewForConfigOrDie(config)
	adapter := adapterFactory(client, config, adapterSpecificArgs)
	controller := newFederationSyncController(client, adapter)
	if minimizeLatency {
		controller.minimizeLatency()
	}
	glog.Infof(fmt.Sprintf("Starting federated sync controller for %s resources", kind))
	controller.Run(stopChan)
}

// newFederationSyncController returns a new sync controller for the given client and type adapter
func newFederationSyncController(client federationclientset.Interface, adapter federatedtypes.FederatedTypeAdapter) *FederationSyncController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.Scheme, v1.EventSource{Component: fmt.Sprintf("federation-%v-controller", adapter.Kind())})

	s := &FederationSyncController{
		reviewDelay:             time.Second * 10,
		clusterAvailableDelay:   time.Second * 20,
		clusterUnavailableDelay: time.Second * 60,
		smallDelay:              time.Second * 3,
		updateTimeout:           time.Second * 30,
		workQueue:               workqueue.New(),
		backoff:                 flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:           recorder,
		adapter:                 adapter,
	}

	// Build delivereres for triggering reconciliations.
	s.deliverer = util.NewDelayingDeliverer()
	s.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on the resource type that should be federated.
	s.store, s.controller = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return adapter.FedList(metav1.NamespaceAll, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return adapter.FedWatch(metav1.NamespaceAll, options)
			},
		},
		adapter.ObjectType(),
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { s.deliverObj(obj, 0, false) }))

	// Federated informer on the resource type in members of federation.
	s.informer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						return adapter.ClusterList(targetClient, metav1.NamespaceAll, options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return adapter.ClusterWatch(targetClient, metav1.NamespaceAll, options)
					},
				},
				adapter.ObjectType(),
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some operation on the target resource type had succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						s.deliverObj(obj, s.reviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the target resources again.
				s.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(s.clusterAvailableDelay))
			},
			// When a cluster becomes unavailable process all the target resources again.
			ClusterUnavailable: func(cluster *federationapi.Cluster, _ []interface{}) {
				s.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(s.clusterUnavailableDelay))
			},
		},
	)

	// Federated updeater along with Create/Update/Delete operations.
	s.updater = util.NewFederatedUpdater(s.informer, adapter.Kind(), s.updateTimeout, s.eventRecorder,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			_, err := adapter.ClusterCreate(client, obj)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			_, err := adapter.ClusterUpdate(client, obj)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			qualifiedName := adapter.QualifiedName(obj)
			orphanDependents := false
			err := adapter.ClusterDelete(client, qualifiedName, &metav1.DeleteOptions{OrphanDependents: &orphanDependents})
			return err
		})

	s.deletionHelper = deletionhelper.NewDeletionHelper(
		s.updateObject,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			return adapter.QualifiedName(obj).String()
		},
		s.informer,
		s.updater,
	)

	return s
}

// minimizeLatency reduces delays and timeouts to make the controller more responsive (useful for testing).
func (s *FederationSyncController) minimizeLatency() {
	s.clusterAvailableDelay = time.Second
	s.clusterUnavailableDelay = time.Second
	s.reviewDelay = 50 * time.Millisecond
	s.smallDelay = 20 * time.Millisecond
	s.updateTimeout = 5 * time.Second
}

// Sends the given updated object to apiserver.
func (s *FederationSyncController) updateObject(obj pkgruntime.Object) (pkgruntime.Object, error) {
	return s.adapter.FedUpdate(obj)
}

func (s *FederationSyncController) Run(stopChan <-chan struct{}) {
	go s.controller.Run(stopChan)
	s.informer.Start()
	s.deliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		s.workQueue.Add(item)
	})
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		s.reconcileOnClusterChange()
	})

	// TODO: Allow multiple workers.
	go wait.Until(s.worker, time.Second, stopChan)

	util.StartBackoffGC(s.backoff, stopChan)

	// Ensure all goroutines are cleaned up when the stop channel closes
	go func() {
		<-stopChan
		s.informer.Stop()
		s.workQueue.ShutDown()
		s.deliverer.Stop()
		s.clusterDeliverer.Stop()
	}()
}

type reconciliationStatus int

const (
	statusAllOK reconciliationStatus = iota
	statusNeedsRecheck
	statusError
	statusNotSynced
)

func (s *FederationSyncController) worker() {
	for {
		obj, quit := s.workQueue.Get()
		if quit {
			return
		}

		item := obj.(*util.DelayingDelivererItem)
		qualifiedName := item.Value.(*federatedtypes.QualifiedName)
		status := s.reconcile(*qualifiedName)
		s.workQueue.Done(item)

		switch status {
		case statusAllOK:
			break
		case statusError:
			s.deliver(*qualifiedName, 0, true)
		case statusNeedsRecheck:
			s.deliver(*qualifiedName, s.reviewDelay, false)
		case statusNotSynced:
			s.deliver(*qualifiedName, s.clusterAvailableDelay, false)
		}
	}
}

func (s *FederationSyncController) deliverObj(obj pkgruntime.Object, delay time.Duration, failed bool) {
	qualifiedName := s.adapter.QualifiedName(obj)
	s.deliver(qualifiedName, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (s *FederationSyncController) deliver(qualifiedName federatedtypes.QualifiedName, delay time.Duration, failed bool) {
	key := qualifiedName.String()
	if failed {
		s.backoff.Next(key, time.Now())
		delay = delay + s.backoff.Get(key)
	} else {
		s.backoff.Reset(key)
	}
	s.deliverer.DeliverAfter(key, &qualifiedName, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
func (s *FederationSyncController) isSynced() bool {
	if !s.informer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := s.informer.GetReadyClusters()
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get ready clusters: %v", err))
		return false
	}
	if !s.informer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconciliation of all target federated resources.
func (s *FederationSyncController) reconcileOnClusterChange() {
	if !s.isSynced() {
		s.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(s.clusterAvailableDelay))
	}
	for _, obj := range s.store.List() {
		qualifiedName := s.adapter.QualifiedName(obj.(pkgruntime.Object))
		s.deliver(qualifiedName, s.smallDelay, false)
	}
}

func (s *FederationSyncController) reconcile(qualifiedName federatedtypes.QualifiedName) reconciliationStatus {
	if !s.isSynced() {
		return statusNotSynced
	}

	kind := s.adapter.Kind()
	key := qualifiedName.String()

	glog.V(4).Infof("Starting to reconcile %v %v", kind, key)
	startTime := time.Now()
	defer glog.V(4).Infof("Finished reconciling %v %v (duration: %v)", kind, key, time.Now().Sub(startTime))

	obj, err := s.objFromCache(kind, key)
	if err != nil {
		return statusError
	}
	if obj == nil {
		return statusAllOK
	}

	meta := s.adapter.ObjectMeta(obj)
	if meta.DeletionTimestamp != nil {
		err := s.delete(obj, kind, qualifiedName)
		if err != nil {
			msg := "Failed to delete %s %q: %v"
			args := []interface{}{kind, qualifiedName, err}
			runtime.HandleError(fmt.Errorf(msg, args...))
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "DeleteFailed", msg, args...)
			return statusError
		}
		return statusAllOK
	}

	glog.V(3).Infof("Ensuring finalizers exist on %s %q", kind, key)
	obj, err = s.deletionHelper.EnsureFinalizers(obj)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to ensure finalizers for %s %q: %v", kind, key, err))
		return statusError
	}

	operationsAccessor := func(adapter federatedtypes.FederatedTypeAdapter, selectedClusters []*federationapi.Cluster, unselectedClusters []*federationapi.Cluster, obj pkgruntime.Object, schedulingInfo interface{}) ([]util.FederatedOperation, error) {
		operations, err := clusterOperations(adapter, selectedClusters, unselectedClusters, obj, key, schedulingInfo, func(clusterName string) (interface{}, bool, error) {
			return s.informer.GetTargetStore().GetByKey(clusterName, key)
		})
		if err != nil {
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "FedClusterOperationsError", "Error obtaining sync operations for %s: %s error: %s", kind, key, err.Error())
		}
		return operations, err
	}

	return syncToClusters(
		s.informer.GetReadyClusters,
		operationsAccessor,
		selectedClusters,
		s.updater.Update,
		s.adapter,
		s.informer,
		obj,
	)
}

func (s *FederationSyncController) objFromCache(kind, key string) (pkgruntime.Object, error) {
	cachedObj, exist, err := s.store.GetByKey(key)
	if err != nil {
		wrappedErr := fmt.Errorf("Failed to query %s store for %q: %v", kind, key, err)
		runtime.HandleError(wrappedErr)
		return nil, err
	}
	if !exist {
		return nil, nil
	}

	// Create a copy before modifying the resource to prevent racing with other readers.
	copiedObj, err := api.Scheme.DeepCopy(cachedObj)
	if err != nil {
		wrappedErr := fmt.Errorf("Error in retrieving %s %q from store: %v", kind, key, err)
		runtime.HandleError(wrappedErr)
		return nil, err
	}
	if !s.adapter.IsExpectedType(copiedObj) {
		err = fmt.Errorf("Object is not the expected type: %v", copiedObj)
		runtime.HandleError(err)
		return nil, err
	}
	return copiedObj.(pkgruntime.Object), nil
}

// delete deletes the given resource or returns error if the deletion was not complete.
func (s *FederationSyncController) delete(obj pkgruntime.Object, kind string, qualifiedName federatedtypes.QualifiedName) error {
	glog.V(3).Infof("Handling deletion of %s %q", kind, qualifiedName)

	// Perform pre-deletion cleanup for the namespace adapter
	namespaceAdapter, ok := s.adapter.(*federatedtypes.NamespaceAdapter)
	if ok {
		var err error
		obj, err = namespaceAdapter.CleanUpNamespace(obj, s.eventRecorder)
		if err != nil {
			return err
		}
	}

	_, err := s.deletionHelper.HandleObjectInUnderlyingClusters(obj)
	if err != nil {
		return err
	}

	err = s.adapter.FedDelete(qualifiedName, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the resource and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

type clustersAccessorFunc func() ([]*federationapi.Cluster, error)
type operationsFunc func(federatedtypes.FederatedTypeAdapter, []*federationapi.Cluster, []*federationapi.Cluster, pkgruntime.Object, interface{}) ([]util.FederatedOperation, error)
type clusterSelectorFunc func(*metav1.ObjectMeta, func(map[string]string, map[string]string) (bool, error), []*federationapi.Cluster) ([]*federationapi.Cluster, []*federationapi.Cluster, error)
type executionFunc func([]util.FederatedOperation) error

// syncToClusters ensures that the state of the given object is synchronized to member clusters.
func syncToClusters(clustersAccessor clustersAccessorFunc, operationsAccessor operationsFunc, selector clusterSelectorFunc, execute executionFunc, adapter federatedtypes.FederatedTypeAdapter, informer util.FederatedInformer, obj pkgruntime.Object) reconciliationStatus {
	kind := adapter.Kind()
	key := federatedtypes.ObjectKey(adapter, obj)

	glog.V(3).Infof("Syncing %s %q in underlying clusters", kind, key)

	clusters, err := clustersAccessor()
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to get cluster list: %v", err))
		return statusNotSynced
	}

	selectedClusters, unselectedClusters, err := selector(adapter.ObjectMeta(obj), clusterselector.SendToCluster, clusters)
	if err != nil {
		return statusError
	}

	var schedulingInfo interface{}
	if adapter.IsSchedulingAdapter() {
		schedulingAdapter, ok := adapter.(federatedtypes.SchedulingAdapter)
		if !ok {
			glog.Fatalf("Adapter for kind %q does not properly implement SchedulingAdapter.", kind)
		}
		schedulingInfo, err = schedulingAdapter.GetSchedule(obj, key, selectedClusters, informer)
		if err != nil {
			runtime.HandleError(fmt.Errorf("adapter.GetSchedule() failed on adapter for %s %q: %v", kind, key, err))
			return statusError
		}
	}

	operations, err := operationsAccessor(adapter, selectedClusters, unselectedClusters, obj, schedulingInfo)
	if err != nil {
		return statusError
	}

	if adapter.IsSchedulingAdapter() {
		schedulingAdapter, ok := adapter.(federatedtypes.SchedulingAdapter)
		if !ok {
			glog.Fatalf("Adapter for kind %q does not properly implement SchedulingAdapter.", kind)
		}
		err = schedulingAdapter.UpdateFederatedStatus(obj, schedulingInfo)
		if err != nil {
			runtime.HandleError(fmt.Errorf("adapter.UpdateFinished() failed on adapter for %s %q: %v", kind, key, err))
			return statusError
		}
	}

	if len(operations) == 0 {
		return statusAllOK
	}

	err = execute(operations)
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to execute updates for %s %q: %v", kind, key, err))
		return statusError
	}

	// Everything is in order but let's be double sure
	return statusNeedsRecheck
}

// selectedClusters filters the provided clusters into two slices, one containing the clusters selected by selector and the other containing the rest of the provided clusters.
func selectedClusters(objMeta *metav1.ObjectMeta, selector func(map[string]string, map[string]string) (bool, error), clusters []*federationapi.Cluster) ([]*federationapi.Cluster, []*federationapi.Cluster, error) {
	selectedClusters := []*federationapi.Cluster{}
	unselectedClusters := []*federationapi.Cluster{}

	for _, cluster := range clusters {
		send, err := selector(cluster.Labels, objMeta.Annotations)
		if err != nil {
			return nil, nil, err
		} else if !send {
			unselectedClusters = append(unselectedClusters, cluster)
		} else {
			selectedClusters = append(selectedClusters, cluster)
		}
	}
	return selectedClusters, unselectedClusters, nil
}

type clusterObjectAccessorFunc func(clusterName string) (interface{}, bool, error)

// clusterOperations returns the list of operations needed to synchronize the state of the given object to the provided clusters
func clusterOperations(adapter federatedtypes.FederatedTypeAdapter, selectedClusters []*federationapi.Cluster, unselectedClusters []*federationapi.Cluster, obj pkgruntime.Object, key string, schedulingInfo interface{}, accessor clusterObjectAccessorFunc) ([]util.FederatedOperation, error) {
	operations := make([]util.FederatedOperation, 0)

	kind := adapter.Kind()
	for _, cluster := range selectedClusters {
		// The data should not be modified.
		desiredObj := adapter.Copy(obj)

		clusterObj, found, err := accessor(cluster.Name)
		if err != nil {
			wrappedErr := fmt.Errorf("Failed to get %s %q from cluster %q: %v", kind, key, cluster.Name, err)
			runtime.HandleError(wrappedErr)
			return nil, wrappedErr
		}

		var scheduleAction federatedtypes.ScheduleAction = federatedtypes.ActionAdd
		if adapter.IsSchedulingAdapter() {
			schedulingAdapter, ok := adapter.(federatedtypes.SchedulingAdapter)
			if !ok {
				err = fmt.Errorf("adapter for kind %s does not properly implement SchedulingAdapter.", kind)
				glog.Fatalf("Error: %v", err)
			}
			var clusterTypedObj pkgruntime.Object = nil
			if clusterObj != nil {
				clusterTypedObj = clusterObj.(pkgruntime.Object)
			}
			desiredObj, scheduleAction, err = schedulingAdapter.ScheduleObject(cluster, clusterTypedObj, desiredObj, schedulingInfo)
			if err != nil {
				runtime.HandleError(err)
				return nil, err
			}
		}

		var operationType util.FederatedOperationType = ""
		if found {
			if scheduleAction == federatedtypes.ActionDelete {
				operationType = util.OperationTypeDelete
			} else {
				clusterObj := clusterObj.(pkgruntime.Object)
				if !adapter.Equivalent(desiredObj, clusterObj) {
					operationType = util.OperationTypeUpdate
				}
			}
		} else if scheduleAction == federatedtypes.ActionAdd {
			operationType = util.OperationTypeAdd
		}

		if len(operationType) > 0 {
			operations = append(operations, util.FederatedOperation{
				Type:        operationType,
				Obj:         desiredObj,
				ClusterName: cluster.Name,
				Key:         key,
			})
		}
	}

	for _, cluster := range unselectedClusters {
		clusterObj, found, err := accessor(cluster.Name)
		if err != nil {
			wrappedErr := fmt.Errorf("Failed to get %s %q from cluster %q: %v", kind, key, cluster.Name, err)
			runtime.HandleError(wrappedErr)
			return nil, wrappedErr
		}
		if found {
			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeDelete,
				Obj:         clusterObj.(pkgruntime.Object),
				ClusterName: cluster.Name,
				Key:         key,
			})
		}
	}

	return operations, nil
}
