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
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
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
	s.updater = util.NewFederatedUpdater(s.informer,
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
			return adapter.ObjectMeta(obj).Name
		},
		s.updateTimeout,
		s.eventRecorder,
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
	go func() {
		<-stopChan
		s.informer.Stop()
		s.workQueue.ShutDown()
	}()
	s.deliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		s.workQueue.Add(item)
	})
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		s.reconcileOnClusterChange()
	})

	// TODO: Allow multiple workers.
	go wait.Until(s.worker, time.Second, stopChan)

	util.StartBackoffGC(s.backoff, stopChan)
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
		namespacedName := item.Value.(*types.NamespacedName)
		status, err := s.reconcile(*namespacedName)
		s.workQueue.Done(item)

		if err != nil {
			glog.Errorf("Error syncing cluster controller: %v", err)
		}
		switch status {
		case statusAllOK:
			break
		case statusError:
			s.deliver(*namespacedName, 0, true)
		case statusNeedsRecheck:
			s.deliver(*namespacedName, s.reviewDelay, false)
		case statusNotSynced:
			s.deliver(*namespacedName, s.clusterAvailableDelay, false)
		default:
			glog.Errorf("Unhandled reconciliation status: %s", status)
			s.deliver(*namespacedName, s.reviewDelay, false)
		}
	}
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

func (s *FederationSyncController) reconcile(namespacedName types.NamespacedName) (reconciliationStatus, error) {
	if !s.isSynced() {
		return statusNotSynced, nil
	}

	key := namespacedName.String()
	kind := s.adapter.Kind()

	glog.V(4).Infof("Starting to reconcile %v %v", kind, key)
	startTime := time.Now()
	defer glog.V(4).Infof("Finished reconciling %v %v (duration: %v)", kind, key, time.Now().Sub(startTime))

	cachedObj, exist, err := s.store.GetByKey(key)
	if err != nil {
		return statusError, fmt.Errorf("failed to query main %s store for %v: %v", kind, key, err)
	}

	if !exist {
		// Not federated, ignoring.
		return statusAllOK, nil
	}

	// Create a copy before modifying the resource to prevent racing
	// with other readers.
	copiedObj, err := api.Scheme.DeepCopy(cachedObj)
	if err != nil {
		return statusError, fmt.Errorf("error in retrieving %s from store: %v", kind, err)
	}
	if !s.adapter.IsExpectedType(copiedObj) {
		return statusError, fmt.Errorf("object is not the expected type: %v", copiedObj)
	}
	obj := copiedObj.(pkgruntime.Object)
	meta := s.adapter.ObjectMeta(obj)

	if meta.DeletionTimestamp != nil {
		if err := s.delete(obj, namespacedName); err != nil {
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "DeleteFailed",
				"%s delete failed: %v", strings.ToTitle(kind), err)
			return statusError, fmt.Errorf("failed to delete %s %s: %v", kind, namespacedName, err)
		}
		return statusAllOK, nil
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for %s: %s",
		kind, namespacedName)
	// Add the required finalizers before creating the resource in underlying clusters.
	obj, err = s.deletionHelper.EnsureFinalizers(obj)
	if err != nil {
		return statusError, fmt.Errorf("failed to ensure delete object from underlying clusters finalizer in %s %s: %v",
			kind, namespacedName, err)
	}

	glog.V(3).Infof("Syncing %s %s in underlying clusters", kind, namespacedName)

	clusters, err := s.informer.GetReadyClusters()
	if err != nil {
		return statusNotSynced, fmt.Errorf("failed to get cluster list: %v", err)
	}

	prepareFunc := s.adapter.PrepareForUpdateFunc()
	var userInfo interface{}
	if prepareFunc != nil {
		userInfo, err = prepareFunc(obj, key, clusters, s.informer)
		if err != nil {
			return statusError, err
		}
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterObj, found, err := s.informer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			return statusError, fmt.Errorf("failed to get %s from %s: %v", key, cluster.Name, err)
		}

		// The data should not be modified.
		desiredObj := s.adapter.Copy(obj)

		// If the adapter has a special way of creating operations, do it.
		updateFunc := s.adapter.UpdateObjectFunc()
		if updateFunc != nil {
			var clusterTypedObj pkgruntime.Object = nil
			if clusterObj != nil {
				clusterTypedObj = clusterObj.(pkgruntime.Object)
			}
			desiredObj, err = updateFunc(cluster, clusterTypedObj, desiredObj, userInfo)
			if err != nil {
				return statusError, nil
			}
		}

		if !found {
			if desiredObj != nil {
				s.eventRecorder.Eventf(obj, api.EventTypeNormal, "CreateInCluster",
					"Creating %s in cluster %s", kind, cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeAdd,
					Obj:         desiredObj,
					ClusterName: cluster.Name,
				})
			}
		} else {
			clusterObj := clusterObj.(pkgruntime.Object)

			// Update existing resource, if needed.
			if !s.adapter.Equivalent(desiredObj, clusterObj) {
				s.eventRecorder.Eventf(obj, api.EventTypeNormal, "UpdateInCluster",
					"Updating %s in cluster %s", kind, cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredObj,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	glog.V(4).Infof("almostFinalObj: %#v", obj)
	updateFinishedFunc := s.adapter.UpdateFinishedFunc()
	if updateFinishedFunc != nil {
		if err = updateFinishedFunc(obj, userInfo); err != nil {
			return statusError, err
		}
	}
	glog.V(4).Infof("finalObj: %#v", obj)

	if len(operations) == 0 {
		// Everything is in order
		return statusAllOK, nil
	}
	err = s.updater.UpdateWithOnError(operations, s.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "UpdateInClusterFailed",
				"%s update in cluster %s failed: %v", strings.ToTitle(kind), op.ClusterName, operror)
		})

	if err != nil {
		return statusError, fmt.Errorf("failed to execute updates for %s: %v", key, err)
	}

	// Evertyhing is in order but let's be double sure
	return statusNeedsRecheck, nil
}

// delete deletes the given resource or returns error if the deletion was not complete.
func (s *FederationSyncController) delete(obj pkgruntime.Object, namespacedName types.NamespacedName) error {
	kind := s.adapter.Kind()
	glog.V(3).Infof("Handling deletion of %s: %v", kind, namespacedName)
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
			return fmt.Errorf("failed to delete %s: %v", kind, err)
		}
	}
	return nil
}
