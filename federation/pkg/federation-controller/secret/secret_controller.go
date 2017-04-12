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

package secret

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/federation/pkg/typeadapters"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
	ControllerName = "secrets"
)

var (
	RequiredResources = []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource("secrets")}
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

	adapter typeadapters.FederatedTypeAdapter
}

// StartSecretController starts a new secret controller
func StartSecretController(config *restclient.Config, stopChan <-chan struct{}, minimizeLatency bool) {
	startFederationSyncController(&typeadapters.SecretAdapter{}, config, stopChan, minimizeLatency)
}

// newSecretController returns a new secret controller
func newSecretController(client federationclientset.Interface) *FederationSyncController {
	return newFederationSyncController(client, typeadapters.NewSecretAdapter(client))
}

// startFederationSyncController starts a new sync controller for the given type adapter
func startFederationSyncController(adapter typeadapters.FederatedTypeAdapter, config *restclient.Config, stopChan <-chan struct{}, minimizeLatency bool) {
	restclient.AddUserAgent(config, fmt.Sprintf("%s-controller", adapter.Kind()))
	client := federationclientset.NewForConfigOrDie(config)
	adapter.SetClient(client)
	controller := newFederationSyncController(client, adapter)
	if minimizeLatency {
		controller.minimizeLatency()
	}
	glog.Infof(fmt.Sprintf("Starting federated sync controller for %s resources", adapter.Kind()))
	controller.Run(stopChan)
}

// newFederationSyncController returns a new sync controller for the given client and type adapter
func newFederationSyncController(client federationclientset.Interface, adapter typeadapters.FederatedTypeAdapter) *FederationSyncController {
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
		s.hasFinalizerFunc,
		s.removeFinalizerFunc,
		s.addFinalizerFunc,
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
	s.reviewDelay = 50 * time.Millisecond
	s.smallDelay = 20 * time.Millisecond
	s.updateTimeout = 5 * time.Second
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (s *FederationSyncController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	meta := s.adapter.ObjectMeta(obj)
	for i := range meta.Finalizers {
		if string(meta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
func (s *FederationSyncController) removeFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	meta := s.adapter.ObjectMeta(obj)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range meta.Finalizers {
		if !deletionhelper.ContainsString(finalizers, meta.Finalizers[i]) {
			newFinalizers = append(newFinalizers, meta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	meta.Finalizers = newFinalizers
	secret, err := s.adapter.FedUpdate(obj)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizers %v from %s %s: %v", finalizers, s.adapter.Kind(), meta.Name, err)
	}
	return secret, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
func (s *FederationSyncController) addFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	meta := s.adapter.ObjectMeta(obj)
	meta.Finalizers = append(meta.Finalizers, finalizers...)
	secret, err := s.adapter.FedUpdate(obj)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to %s %s: %v", finalizers, s.adapter.Kind(), meta.Name, err)
	}
	return secret, nil
}

func (s *FederationSyncController) Run(stopChan <-chan struct{}) {
	go s.controller.Run(stopChan)
	s.informer.Start()
	go func() {
		<-stopChan
		s.informer.Stop()
	}()
	s.deliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		namespacedName := item.Value.(*types.NamespacedName)
		s.reconcile(*namespacedName)
	})
	s.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		s.reconcileOnClusterChange()
	})
	util.StartBackoffGC(s.backoff, stopChan)
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

func (s *FederationSyncController) reconcile(namespacedName types.NamespacedName) {
	if !s.isSynced() {
		s.deliver(namespacedName, s.clusterAvailableDelay, false)
		return
	}

	key := namespacedName.String()
	kind := s.adapter.Kind()
	cachedObj, exist, err := s.store.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main %s store for %v: %v", kind, key, err)
		s.deliver(namespacedName, 0, true)
		return
	}

	if !exist {
		// Not federated, ignoring.
		return
	}

	// Create a copy before modifying the resource to prevent racing
	// with other readers.
	copiedObj, err := api.Scheme.DeepCopy(cachedObj)
	if err != nil {
		glog.Errorf("Error in retrieving %s from store: %v", kind, err)
		s.deliver(namespacedName, 0, true)
		return
	}
	if !s.adapter.IsExpectedType(copiedObj) {
		glog.Errorf("Object is not the expected type: %v", copiedObj)
		s.deliver(namespacedName, 0, true)
		return
	}
	obj := copiedObj.(pkgruntime.Object)
	meta := s.adapter.ObjectMeta(obj)

	if meta.DeletionTimestamp != nil {
		if err := s.delete(obj, namespacedName); err != nil {
			glog.Errorf("Failed to delete %s %s: %v", kind, namespacedName, err)
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "DeleteFailed",
				"%s delete failed: %v", strings.ToTitle(kind), err)
			s.deliver(namespacedName, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for %s: %s",
		kind, namespacedName)
	// Add the required finalizers before creating the resource in underlying clusters.
	obj, err = s.deletionHelper.EnsureFinalizers(obj)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in %s %s: %v",
			kind, namespacedName, err)
		s.deliver(namespacedName, 0, false)
		return
	}

	glog.V(3).Infof("Syncing %s %s in underlying clusters", kind, namespacedName)

	clusters, err := s.informer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		s.deliver(namespacedName, s.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterObj, found, err := s.informer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", key, cluster.Name, err)
			s.deliver(namespacedName, 0, true)
			return
		}

		// The data should not be modified.
		desiredObj := s.adapter.Copy(obj)

		if !found {
			s.eventRecorder.Eventf(obj, api.EventTypeNormal, "CreateInCluster",
				"Creating %s in cluster %s", kind, cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredObj,
				ClusterName: cluster.Name,
			})
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

	if len(operations) == 0 {
		// Everything is in order
		return
	}
	err = s.updater.UpdateWithOnError(operations, s.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			s.eventRecorder.Eventf(obj, api.EventTypeWarning, "UpdateInClusterFailed",
				"%s update in cluster %s failed: %v", strings.ToTitle(kind), op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v", key, err)
		s.deliver(namespacedName, 0, true)
		return
	}

	// Evertyhing is in order but lets be double sure
	s.deliver(namespacedName, s.reviewDelay, false)
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
