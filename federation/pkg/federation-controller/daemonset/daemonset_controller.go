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

package daemonset

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
	ControllerName = "daemonsets"
)

var (
	RequiredResources = []schema.GroupVersionResource{extensionsv1.SchemeGroupVersion.WithResource("daemonsets")}
)

type DaemonSetController struct {
	// For triggering single daemonset reconciliation. This is used when there is an
	// add/update/delete operation on a daemonset in either federated API server or
	// in some member of the federation.
	daemonsetDeliverer *util.DelayingDeliverer

	// For triggering all daemonsets reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains daemonsets present in members of federation.
	daemonsetFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of daemonsets that should be federated.
	daemonsetInformerStore cache.Store
	// Informer controller for daemonsets that should be federated.
	daemonsetInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for daemonsets
	daemonsetBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	deletionHelper *deletionhelper.DeletionHelper

	daemonsetReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewDaemonSetController returns a new daemonset controller
func NewDaemonSetController(client federationclientset.Interface) *DaemonSetController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-daemonset-controller"})

	daemonsetcontroller := &DaemonSetController{
		federatedApiClient:    client,
		daemonsetReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		daemonsetBackoff:      flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build deliverers for triggering reconciliations.
	daemonsetcontroller.daemonsetDeliverer = util.NewDelayingDeliverer()
	daemonsetcontroller.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer in federated API servers on daemonsets that should be federated.
	daemonsetcontroller.daemonsetInformerStore, daemonsetcontroller.daemonsetInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return client.Extensions().DaemonSets(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Extensions().DaemonSets(metav1.NamespaceAll).Watch(options)
			},
		},
		&extensionsv1.DaemonSet{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { daemonsetcontroller.deliverDaemonSetObj(obj, 0, false) }))

	// Federated informer on daemonsets in members of federation.
	daemonsetcontroller.daemonsetFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Extensions().DaemonSets(metav1.NamespaceAll).List(options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return targetClient.Extensions().DaemonSets(metav1.NamespaceAll).Watch(options)
					},
				},
				&extensionsv1.DaemonSet{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some daemonset operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						daemonsetcontroller.deliverDaemonSetObj(obj, daemonsetcontroller.daemonsetReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the daemonsets again.
				daemonsetcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(daemonsetcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	daemonsetcontroller.federatedUpdater = util.NewFederatedUpdater(daemonsetcontroller.daemonsetFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			daemonset := obj.(*extensionsv1.DaemonSet)
			glog.V(4).Infof("Attempting to create daemonset: %s/%s", daemonset.Namespace, daemonset.Name)
			_, err := client.Extensions().DaemonSets(daemonset.Namespace).Create(daemonset)
			if err != nil {
				glog.Errorf("Error creating daemonset %s/%s/: %v", daemonset.Namespace, daemonset.Name, err)
			} else {
				glog.V(4).Infof("Successfully created daemonset %s/%s", daemonset.Namespace, daemonset.Name)
			}
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			daemonset := obj.(*extensionsv1.DaemonSet)
			glog.V(4).Infof("Attempting to update daemonset: %s/%s", daemonset.Namespace, daemonset.Name)
			_, err := client.Extensions().DaemonSets(daemonset.Namespace).Update(daemonset)
			if err != nil {
				glog.Errorf("Error updating daemonset %s/%s/: %v", daemonset.Namespace, daemonset.Name, err)
			} else {
				glog.V(4).Infof("Successfully updating daemonset %s/%s", daemonset.Namespace, daemonset.Name)
			}
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			daemonset := obj.(*extensionsv1.DaemonSet)
			glog.V(4).Infof("Attempting to delete daemonset: %s/%s", daemonset.Namespace, daemonset.Name)
			err := client.Extensions().DaemonSets(daemonset.Namespace).Delete(daemonset.Name, &metav1.DeleteOptions{})
			if err != nil {
				glog.Errorf("Error deleting daemonset %s/%s/: %v", daemonset.Namespace, daemonset.Name, err)
			} else {
				glog.V(4).Infof("Successfully deleting daemonset %s/%s", daemonset.Namespace, daemonset.Name)
			}
			return err
		})

	daemonsetcontroller.deletionHelper = deletionhelper.NewDeletionHelper(
		daemonsetcontroller.hasFinalizerFunc,
		daemonsetcontroller.removeFinalizerFunc,
		daemonsetcontroller.addFinalizerFunc,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			daemonset := obj.(*extensionsv1.DaemonSet)
			return daemonset.Name
		},
		daemonsetcontroller.updateTimeout,
		daemonsetcontroller.eventRecorder,
		daemonsetcontroller.daemonsetFederatedInformer,
		daemonsetcontroller.federatedUpdater,
	)

	return daemonsetcontroller
}

// Returns true if the given object has the given finalizer in its ObjectMeta.
func (daemonsetcontroller *DaemonSetController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	daemonset := obj.(*extensionsv1.DaemonSet)
	for i := range daemonset.ObjectMeta.Finalizers {
		if string(daemonset.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// Removes the finalizer from the given objects ObjectMeta.
// Assumes that the given object is a daemonset.
func (daemonsetcontroller *DaemonSetController) removeFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range daemonset.ObjectMeta.Finalizers {
		if string(daemonset.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, daemonset.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	daemonset.ObjectMeta.Finalizers = newFinalizers
	daemonset, err := daemonsetcontroller.federatedApiClient.Extensions().DaemonSets(daemonset.Namespace).Update(daemonset)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from daemonset %s: %v", finalizer, daemonset.Name, err)
	}
	return daemonset, nil
}

// Adds the given finalizers to the given objects ObjectMeta.
// Assumes that the given object is a daemonset.
func (daemonsetcontroller *DaemonSetController) addFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	daemonset.ObjectMeta.Finalizers = append(daemonset.ObjectMeta.Finalizers, finalizers...)
	daemonset, err := daemonsetcontroller.federatedApiClient.Extensions().DaemonSets(daemonset.Namespace).Update(daemonset)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to daemonset %s: %v", finalizers, daemonset.Name, err)
	}
	return daemonset, nil
}

func (daemonsetcontroller *DaemonSetController) Run(stopChan <-chan struct{}) {
	glog.V(1).Infof("Starting daemonset controller")
	go daemonsetcontroller.daemonsetInformerController.Run(stopChan)

	glog.V(1).Infof("Starting daemonset federated informer")
	daemonsetcontroller.daemonsetFederatedInformer.Start()
	go func() {
		<-stopChan
		daemonsetcontroller.daemonsetFederatedInformer.Stop()
	}()
	glog.V(1).Infof("Starting daemonset deliverers")
	daemonsetcontroller.daemonsetDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		daemonset := item.Value.(*types.NamespacedName)
		glog.V(4).Infof("Trigerring reconciliation of daemonset %s", daemonset.String())
		daemonsetcontroller.reconcileDaemonSet(daemonset.Namespace, daemonset.Name)
	})
	daemonsetcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		glog.V(4).Infof("Triggering reconciliation of all daemonsets")
		daemonsetcontroller.reconcileDaemonSetsOnClusterChange()
	})
	util.StartBackoffGC(daemonsetcontroller.daemonsetBackoff, stopChan)
}

func getDaemonSetKey(namespace, name string) string {
	return types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}.String()
}

func (daemonsetcontroller *DaemonSetController) deliverDaemonSetObj(obj interface{}, delay time.Duration, failed bool) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	daemonsetcontroller.deliverDaemonSet(daemonset.Namespace, daemonset.Name, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (daemonsetcontroller *DaemonSetController) deliverDaemonSet(namespace string, name string, delay time.Duration, failed bool) {
	key := getDaemonSetKey(namespace, name)
	if failed {
		daemonsetcontroller.daemonsetBackoff.Next(key, time.Now())
		delay = delay + daemonsetcontroller.daemonsetBackoff.Get(key)
	} else {
		daemonsetcontroller.daemonsetBackoff.Reset(key)
	}
	daemonsetcontroller.daemonsetDeliverer.DeliverAfter(key,
		&types.NamespacedName{Namespace: namespace, Name: name}, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
func (daemonsetcontroller *DaemonSetController) isSynced() bool {
	if !daemonsetcontroller.daemonsetFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := daemonsetcontroller.daemonsetFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !daemonsetcontroller.daemonsetFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconciliation of all federated daemonsets.
func (daemonsetcontroller *DaemonSetController) reconcileDaemonSetsOnClusterChange() {
	if !daemonsetcontroller.isSynced() {
		daemonsetcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(daemonsetcontroller.clusterAvailableDelay))
	}
	for _, obj := range daemonsetcontroller.daemonsetInformerStore.List() {
		daemonset := obj.(*extensionsv1.DaemonSet)
		daemonsetcontroller.deliverDaemonSet(daemonset.Namespace, daemonset.Name, daemonsetcontroller.smallDelay, false)
	}
}

func (daemonsetcontroller *DaemonSetController) reconcileDaemonSet(namespace string, daemonsetName string) {
	glog.V(4).Infof("Reconciling daemonset %s/%s", namespace, daemonsetName)

	if !daemonsetcontroller.isSynced() {
		glog.V(4).Infof("Daemonset controller is not synced")
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, daemonsetcontroller.clusterAvailableDelay, false)
		return
	}

	key := getDaemonSetKey(namespace, daemonsetName)
	baseDaemonSetObjFromStore, exist, err := daemonsetcontroller.daemonsetInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main daemonset store for %v: %v", key, err)
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, true)
		return
	}

	if !exist {
		glog.V(4).Infof("Skipping daemonset %s/%s - not federated", namespace, daemonsetName)
		// Not federated daemonset, ignoring.
		return
	}
	baseDaemonSetObj, err := api.Scheme.DeepCopy(baseDaemonSetObjFromStore)
	baseDaemonSet, ok := baseDaemonSetObj.(*extensionsv1.DaemonSet)
	if err != nil || !ok {
		glog.Errorf("Error in retrieving obj %s from store: %v, %v", daemonsetName, ok, err)
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, true)
		return
	}
	if baseDaemonSet.DeletionTimestamp != nil {
		if err := daemonsetcontroller.delete(baseDaemonSet); err != nil {
			glog.Errorf("Failed to delete %s: %v", daemonsetName, err)
			daemonsetcontroller.eventRecorder.Eventf(baseDaemonSet, api.EventTypeNormal, "DeleteFailed",
				"DaemonSet delete failed: %v", err)
			daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for daemonset: %s",
		baseDaemonSet.Name)
	// Add the required finalizers before creating a daemonset in underlying clusters.
	updatedDaemonSetObj, err := daemonsetcontroller.deletionHelper.EnsureFinalizers(baseDaemonSet)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in daemonset %s: %v",
			baseDaemonSet.Name, err)
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, false)
		return
	}
	baseDaemonSet = updatedDaemonSetObj.(*extensionsv1.DaemonSet)

	glog.V(3).Infof("Syncing daemonset %s in underlying clusters", baseDaemonSet.Name)

	clusters, err := daemonsetcontroller.daemonsetFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v", err)
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, daemonsetcontroller.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterDaemonSetObj, found, err := daemonsetcontroller.daemonsetFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v", key, cluster.Name, err)
			daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, true)
			return
		}

		// Do not modify. Otherwise make a deep copy.
		desiredDaemonSet := &extensionsv1.DaemonSet{
			ObjectMeta: util.DeepCopyRelevantObjectMeta(baseDaemonSet.ObjectMeta),
			Spec:       *(util.DeepCopyApiTypeOrPanic(&baseDaemonSet.Spec).(*extensionsv1.DaemonSetSpec)),
		}

		if !found {
			glog.V(4).Infof("Creating daemonset %s/%s in cluster %s", namespace, daemonsetName, cluster.Name)

			daemonsetcontroller.eventRecorder.Eventf(baseDaemonSet, api.EventTypeNormal, "CreateInCluster",
				"Creating daemonset in cluster %s", cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredDaemonSet,
				ClusterName: cluster.Name,
			})
		} else {
			clusterDaemonSet := clusterDaemonSetObj.(*extensionsv1.DaemonSet)

			// Update existing daemonset, if needed.
			if !util.ObjectMetaEquivalent(desiredDaemonSet.ObjectMeta, clusterDaemonSet.ObjectMeta) ||
				!reflect.DeepEqual(desiredDaemonSet.Spec, clusterDaemonSet.Spec) {

				glog.V(4).Infof("Upadting daemonset %s/%s in cluster %s", namespace, daemonsetName, cluster.Name)
				daemonsetcontroller.eventRecorder.Eventf(baseDaemonSet, api.EventTypeNormal, "UpdateInCluster",
					"Updating daemonset in cluster %s", cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredDaemonSet,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		glog.V(4).Infof("No operation needed for %s/%s", namespace, daemonsetName)
		// Everything is in order
		return
	}
	err = daemonsetcontroller.federatedUpdater.UpdateWithOnError(operations, daemonsetcontroller.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			daemonsetcontroller.eventRecorder.Eventf(baseDaemonSet, api.EventTypeNormal, "UpdateInClusterFailed",
				"DaemonSet update in cluster %s failed: %v", op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v, retrying shortly", key, err)
		daemonsetcontroller.deliverDaemonSet(namespace, daemonsetName, 0, true)
		return
	}
}

// delete deletes the given daemonset or returns error if the deletion was not complete.
func (daemonsetcontroller *DaemonSetController) delete(daemonset *extensionsv1.DaemonSet) error {
	glog.V(3).Infof("Handling deletion of daemonset: %v", *daemonset)
	_, err := daemonsetcontroller.deletionHelper.HandleObjectInUnderlyingClusters(daemonset)
	if err != nil {
		return err
	}

	err = daemonsetcontroller.federatedApiClient.Extensions().DaemonSets(daemonset.Namespace).Delete(daemonset.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of daemonset finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the daemonset and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete daemonset: %v", err)
		}
	}
	return nil
}
