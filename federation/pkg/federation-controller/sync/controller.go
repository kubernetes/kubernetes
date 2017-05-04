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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
)

type reconcileStatus int

const (
	reconciled reconcileStatus = iota
	redeliver
	redeliverForFailure
	redeliverForClusterReadiness
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

	// Backoff manager
	backoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	reviewDelay           time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration

	adapter federatedtypes.FederatedTypeAdapter
}

// StartFederationSyncController starts a new sync controller for a type adapter
func StartFederationSyncController(kind string, adapterFactory federatedtypes.AdapterFactory, config *restclient.Config, stopChan <-chan struct{}, minimizeLatency bool) {
	restclient.AddUserAgent(config, fmt.Sprintf("%s-controller", kind))
	client := federationclientset.NewForConfigOrDie(config)
	adapter := adapterFactory(client)
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
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: fmt.Sprintf("federated-%v-controller", adapter.Kind())})

	s := &FederationSyncController{
		reviewDelay:           time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		backoff:               flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
		adapter:               adapter,
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
			namespacedName := adapter.NamespacedName(obj)
			orphanDependents := false
			err := adapter.ClusterDelete(client, namespacedName, &metav1.DeleteOptions{OrphanDependents: &orphanDependents})
			return err
		})

	s.deletionHelper = deletionhelper.NewDeletionHelper(
		s.updateObject,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			return adapter.NamespacedName(obj).String()
		},
		s.informer,
		s.updater,
	)

	return s
}

// minimizeLatency reduces delays and timeouts to make the controller more responsive (useful for testing).
func (s *FederationSyncController) minimizeLatency() {
	s.clusterAvailableDelay = time.Second
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
		namespacedName := *item.Value.(*types.NamespacedName)
		status := s.reconcile(namespacedName)
		switch status {
		case redeliver:
			s.deliver(namespacedName, 0, false)
		case redeliverForFailure:
			s.deliver(namespacedName, 0, true)
		case redeliverForClusterReadiness:
			s.deliver(namespacedName, s.clusterAvailableDelay, false)
		}
	})
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		s.reconcileOnClusterChange()
	})
	util.StartBackoffGC(s.backoff, stopChan)
	// Ensure all goroutines are cleaned up when the stop channel closes
	go func() {
		<-stopChan
		s.informer.Stop()
		s.deliverer.Stop()
		s.clusterDeliverer.Stop()
	}()
}

func (s *FederationSyncController) deliverObj(obj pkgruntime.Object, delay time.Duration, failed bool) {
	namespacedName := s.adapter.NamespacedName(obj)
	s.deliver(namespacedName, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (s *FederationSyncController) deliver(namespacedName types.NamespacedName, delay time.Duration, failed bool) {
	key := namespacedName.String()
	if failed {
		s.backoff.Next(key, time.Now())
		delay = delay + s.backoff.Get(key)
	} else {
		s.backoff.Reset(key)
	}
	s.deliverer.DeliverAfter(key, &namespacedName, delay)
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
		glog.Errorf("Failed to get ready clusters: %v", err)
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
		namespacedName := s.adapter.NamespacedName(obj.(pkgruntime.Object))
		s.deliver(namespacedName, s.smallDelay, false)
	}
}

func (s *FederationSyncController) reconcile(namespacedName types.NamespacedName) reconcileStatus {
	if !s.isSynced() {
		return redeliverForClusterReadiness
	}

	kind := s.adapter.Kind()
	key := namespacedName.String()

	obj, err := s.objFromCache(kind, key)
	if err != nil {
		glog.Error(err)
		return redeliverForFailure
	}
	if obj == nil {
		return reconciled
	}

	meta := s.adapter.ObjectMeta(obj)
	if meta.DeletionTimestamp != nil {
		return s.deleteWithStatus(obj, kind, namespacedName)
	}

	glog.V(3).Infof("Ensuring finalizers for %s %q", kind, key)
	obj, err = s.deletionHelper.EnsureFinalizers(obj)
	if err != nil {
		glog.Errorf("Failed to ensure finalizers for %s %q: %v", kind, key, err)
		return redeliver
	}

	return syncToClusters(
		s.informer.GetReadyClusters,
		func(adapter federatedtypes.FederatedTypeAdapter, clusters []*federationapi.Cluster, obj pkgruntime.Object) ([]util.FederatedOperation, error) {
			return clusterOperations(adapter, clusters, obj, func(clusterName string) (interface{}, bool, error) {
				return s.informer.GetTargetStore().GetByKey(clusterName, key)
			})
		},
		s.updater.Update,
		s.adapter,
		obj,
	)
}

func (s *FederationSyncController) objFromCache(kind, key string) (pkgruntime.Object, error) {
	cachedObj, exist, err := s.store.GetByKey(key)
	if err != nil {
		return nil, fmt.Errorf("Failed to query %s store for %q: %v", kind, key, err)
	}
	if !exist {
		return nil, nil
	}

	// Create a copy before modifying the resource to prevent racing
	// with other readers.
	copiedObj, err := api.Scheme.DeepCopy(cachedObj)
	if err != nil {
		return nil, fmt.Errorf("Error in retrieving %s %q from store: %v", kind, key, err)
	}
	if !s.adapter.IsExpectedType(copiedObj) {
		return nil, fmt.Errorf("Object is not the expected type: %v", copiedObj)
	}
	return copiedObj.(pkgruntime.Object), nil
}

func (s *FederationSyncController) deleteWithStatus(obj pkgruntime.Object, kind string, namespacedName types.NamespacedName) reconcileStatus {
	if err := s.delete(obj, kind, namespacedName); err != nil {
		msg := "Failed to delete %s %q: %v"
		args := []interface{}{kind, namespacedName, err}
		glog.Errorf(msg, args...)
		s.eventRecorder.Eventf(obj, api.EventTypeWarning, "DeleteFailed", msg, args...)
		return redeliverForFailure
	}
	return reconciled
}

// delete deletes the given resource or returns error if the deletion was not complete.
func (s *FederationSyncController) delete(obj pkgruntime.Object, kind string, namespacedName types.NamespacedName) error {
	glog.V(3).Infof("Handling deletion of %s %q", kind, namespacedName)
	_, err := s.deletionHelper.HandleObjectInUnderlyingClusters(obj)
	if err != nil {
		return err
	}

	err = s.adapter.FedDelete(namespacedName, nil)
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
type operationsFunc func(federatedtypes.FederatedTypeAdapter, []*federationapi.Cluster, pkgruntime.Object) ([]util.FederatedOperation, error)
type executionFunc func([]util.FederatedOperation) error

// syncToClusters ensures that the state of the given object is synchronized to member clusters.
func syncToClusters(clustersAccessor clustersAccessorFunc, operationsAccessor operationsFunc, execute executionFunc, adapter federatedtypes.FederatedTypeAdapter, obj pkgruntime.Object) reconcileStatus {
	kind := adapter.Kind()
	key := federatedtypes.ObjectKey(adapter, obj)

	glog.V(3).Infof("Syncing %s %q in underlying clusters", kind, key)

	clusters, err := clustersAccessor()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		return redeliverForClusterReadiness
	}

	operations, err := operationsAccessor(adapter, clusters, obj)
	if err != nil {
		glog.Error(err)
		return redeliverForFailure
	}
	if len(operations) == 0 {
		return reconciled
	}

	err = execute(operations)
	if err != nil {
		glog.Errorf("Failed to execute updates for %s %q: %v", kind, key, err)
		return redeliverForFailure
	}

	return reconciled
}

type clusterObjectAccessorFunc func(clusterName string) (interface{}, bool, error)

// clusterOperations returns the list of operations needed to synchronize the state of the given object to the provided clusters
func clusterOperations(adapter federatedtypes.FederatedTypeAdapter, clusters []*federationapi.Cluster, obj pkgruntime.Object, accessor clusterObjectAccessorFunc) ([]util.FederatedOperation, error) {
	key := federatedtypes.ObjectKey(adapter, obj)
	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterObj, found, err := accessor(cluster.Name)
		if err != nil {
			return nil, fmt.Errorf("Failed to get %s %q from cluster %q: %v", adapter.Kind(), key, cluster.Name, err)
		}
		// The data should not be modified.
		desiredObj := adapter.Copy(obj)

		var operationType util.FederatedOperationType = ""
		if found {
			clusterObj := clusterObj.(pkgruntime.Object)
			if !adapter.Equivalent(desiredObj, clusterObj) {
				operationType = util.OperationTypeUpdate
			}
		} else {
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
	return operations, nil
}
