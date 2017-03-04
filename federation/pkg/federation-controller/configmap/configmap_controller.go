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

package configmap

import (
	"fmt"
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
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	allClustersKey = "ALL_CLUSTERS"
	ControllerName = "configmaps"
)

var (
	RequiredResources = []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource("configmaps")}
)

type ConfigMapController struct {
	// For triggering single configmap reconciliation. This is used when there is an
	// add/update/delete operation on a configmap in either federated API server or
	// in some member of the federation.
	configmapDeliverer *util.DelayingDeliverer

	// For triggering all configmaps reconciliation. This is used when
	// a new cluster becomes available.
	clusterDeliverer *util.DelayingDeliverer

	// Contains configmaps present in members of federation.
	configmapFederatedInformer util.FederatedInformer
	// For updating members of federation.
	federatedUpdater util.FederatedUpdater
	// Definitions of configmaps that should be federated.
	configmapInformerStore cache.Store
	// Informer controller for configmaps that should be federated.
	configmapInformerController cache.Controller

	// Client to federated api server.
	federatedApiClient federationclientset.Interface

	// Backoff manager for configmaps
	configmapBackoff *flowcontrol.Backoff

	// For events
	eventRecorder record.EventRecorder

	// Finalizers
	deletionHelper *deletionhelper.DeletionHelper

	configmapReviewDelay  time.Duration
	clusterAvailableDelay time.Duration
	smallDelay            time.Duration
	updateTimeout         time.Duration
}

// NewConfigMapController returns a new configmap controller
func NewConfigMapController(client federationclientset.Interface) *ConfigMapController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(client))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-configmaps-controller"})

	configmapcontroller := &ConfigMapController{
		federatedApiClient:    client,
		configmapReviewDelay:  time.Second * 10,
		clusterAvailableDelay: time.Second * 20,
		smallDelay:            time.Second * 3,
		updateTimeout:         time.Second * 30,
		configmapBackoff:      flowcontrol.NewBackOff(5*time.Second, time.Minute),
		eventRecorder:         recorder,
	}

	// Build delivereres for triggering reconciliations.
	configmapcontroller.configmapDeliverer = util.NewDelayingDeliverer()
	configmapcontroller.clusterDeliverer = util.NewDelayingDeliverer()

	// Start informer on federated API servers on configmaps that should be federated.
	configmapcontroller.configmapInformerStore, configmapcontroller.configmapInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
				return client.Core().ConfigMaps(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Core().ConfigMaps(metav1.NamespaceAll).Watch(options)
			},
		},
		&apiv1.ConfigMap{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { configmapcontroller.deliverConfigMapObj(obj, 0, false) }))

	// Federated informer on configmaps in members of federation.
	configmapcontroller.configmapFederatedInformer = util.NewFederatedInformer(
		client,
		func(cluster *federationapi.Cluster, targetClient kubeclientset.Interface) (cache.Store, cache.Controller) {
			return cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (pkgruntime.Object, error) {
						return targetClient.Core().ConfigMaps(metav1.NamespaceAll).List(options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return targetClient.Core().ConfigMaps(metav1.NamespaceAll).Watch(options)
					},
				},
				&apiv1.ConfigMap{},
				controller.NoResyncPeriodFunc(),
				// Trigger reconciliation whenever something in federated cluster is changed. In most cases it
				// would be just confirmation that some configmap operation succeeded.
				util.NewTriggerOnAllChanges(
					func(obj pkgruntime.Object) {
						configmapcontroller.deliverConfigMapObj(obj, configmapcontroller.configmapReviewDelay, false)
					},
				))
		},

		&util.ClusterLifecycleHandlerFuncs{
			ClusterAvailable: func(cluster *federationapi.Cluster) {
				// When new cluster becomes available process all the configmaps again.
				configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
			},
		},
	)

	// Federated updater along with Create/Update/Delete operations.
	configmapcontroller.federatedUpdater = util.NewFederatedUpdater(configmapcontroller.configmapFederatedInformer,
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Create(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			_, err := client.Core().ConfigMaps(configmap.Namespace).Update(configmap)
			return err
		},
		func(client kubeclientset.Interface, obj pkgruntime.Object) error {
			configmap := obj.(*apiv1.ConfigMap)
			err := client.Core().ConfigMaps(configmap.Namespace).Delete(configmap.Name, &metav1.DeleteOptions{})
			return err
		})

	configmapcontroller.deletionHelper = deletionhelper.NewDeletionHelper(
		configmapcontroller.hasFinalizerFunc,
		configmapcontroller.removeFinalizerFunc,
		configmapcontroller.addFinalizerFunc,
		// objNameFunc
		func(obj pkgruntime.Object) string {
			configmap := obj.(*apiv1.ConfigMap)
			return configmap.Name
		},
		configmapcontroller.updateTimeout,
		configmapcontroller.eventRecorder,
		configmapcontroller.configmapFederatedInformer,
		configmapcontroller.federatedUpdater,
	)

	return configmapcontroller
}

// hasFinalizerFunc returns true if the given object has the given finalizer in its ObjectMeta.
func (configmapcontroller *ConfigMapController) hasFinalizerFunc(obj pkgruntime.Object, finalizer string) bool {
	configmap := obj.(*apiv1.ConfigMap)
	for i := range configmap.ObjectMeta.Finalizers {
		if string(configmap.ObjectMeta.Finalizers[i]) == finalizer {
			return true
		}
	}
	return false
}

// removeFinalizerFunc removes the finalizer from the given objects ObjectMeta. Assumes that the given object is a configmap.
func (configmapcontroller *ConfigMapController) removeFinalizerFunc(obj pkgruntime.Object, finalizer string) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	newFinalizers := []string{}
	hasFinalizer := false
	for i := range configmap.ObjectMeta.Finalizers {
		if string(configmap.ObjectMeta.Finalizers[i]) != finalizer {
			newFinalizers = append(newFinalizers, configmap.ObjectMeta.Finalizers[i])
		} else {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		// Nothing to do.
		return obj, nil
	}
	configmap.ObjectMeta.Finalizers = newFinalizers
	configmap, err := configmapcontroller.federatedApiClient.Core().ConfigMaps(configmap.Namespace).Update(configmap)
	if err != nil {
		return nil, fmt.Errorf("failed to remove finalizer %s from configmap %s: %v", finalizer, configmap.Name, err)
	}
	return configmap, nil
}

// addFinalizerFunc adds the given finalizer to the given objects ObjectMeta. Assumes that the given object is a configmap.
func (configmapcontroller *ConfigMapController) addFinalizerFunc(obj pkgruntime.Object, finalizers []string) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	configmap.ObjectMeta.Finalizers = append(configmap.ObjectMeta.Finalizers, finalizers...)
	configmap, err := configmapcontroller.federatedApiClient.Core().ConfigMaps(configmap.Namespace).Update(configmap)
	if err != nil {
		return nil, fmt.Errorf("failed to add finalizers %v to configmap %s: %v", finalizers, configmap.Name, err)
	}
	return configmap, nil
}

func (configmapcontroller *ConfigMapController) Run(stopChan <-chan struct{}) {
	go configmapcontroller.configmapInformerController.Run(stopChan)
	configmapcontroller.configmapFederatedInformer.Start()
	go func() {
		<-stopChan
		configmapcontroller.configmapFederatedInformer.Stop()
	}()
	configmapcontroller.configmapDeliverer.StartWithHandler(func(item *util.DelayingDelivererItem) {
		configmap := item.Value.(*types.NamespacedName)
		configmapcontroller.reconcileConfigMap(*configmap)
	})
	configmapcontroller.clusterDeliverer.StartWithHandler(func(_ *util.DelayingDelivererItem) {
		configmapcontroller.reconcileConfigMapsOnClusterChange()
	})
	util.StartBackoffGC(configmapcontroller.configmapBackoff, stopChan)
}

func (configmapcontroller *ConfigMapController) deliverConfigMapObj(obj interface{}, delay time.Duration, failed bool) {
	configmap := obj.(*apiv1.ConfigMap)
	configmapcontroller.deliverConfigMap(types.NamespacedName{Namespace: configmap.Namespace, Name: configmap.Name}, delay, failed)
}

// Adds backoff to delay if this delivery is related to some failure. Resets backoff if there was no failure.
func (configmapcontroller *ConfigMapController) deliverConfigMap(configmap types.NamespacedName, delay time.Duration, failed bool) {
	key := configmap.String()
	if failed {
		configmapcontroller.configmapBackoff.Next(key, time.Now())
		delay = delay + configmapcontroller.configmapBackoff.Get(key)
	} else {
		configmapcontroller.configmapBackoff.Reset(key)
	}
	configmapcontroller.configmapDeliverer.DeliverAfter(key, &configmap, delay)
}

// Check whether all data stores are in sync. False is returned if any of the informer/stores is not yet
// synced with the corresponding api server.
func (configmapcontroller *ConfigMapController) isSynced() bool {
	if !configmapcontroller.configmapFederatedInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := configmapcontroller.configmapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !configmapcontroller.configmapFederatedInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}
	return true
}

// The function triggers reconciliation of all federated configmaps.
func (configmapcontroller *ConfigMapController) reconcileConfigMapsOnClusterChange() {
	if !configmapcontroller.isSynced() {
		glog.V(4).Infof("Configmap controller not synced")
		configmapcontroller.clusterDeliverer.DeliverAt(allClustersKey, nil, time.Now().Add(configmapcontroller.clusterAvailableDelay))
	}
	for _, obj := range configmapcontroller.configmapInformerStore.List() {
		configmap := obj.(*apiv1.ConfigMap)
		configmapcontroller.deliverConfigMap(types.NamespacedName{Namespace: configmap.Namespace, Name: configmap.Name},
			configmapcontroller.smallDelay, false)
	}
}

func (configmapcontroller *ConfigMapController) reconcileConfigMap(configmap types.NamespacedName) {

	if !configmapcontroller.isSynced() {
		glog.V(4).Infof("Configmap controller not synced")
		configmapcontroller.deliverConfigMap(configmap, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	key := configmap.String()
	baseConfigMapObj, exist, err := configmapcontroller.configmapInformerStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Failed to query main configmap store for %v: %v", key, err)
		configmapcontroller.deliverConfigMap(configmap, 0, true)
		return
	}

	if !exist {
		// Not federated configmap, ignoring.
		glog.V(8).Infof("Skipping not federated config map: %s", key)
		return
	}
	baseConfigMap := baseConfigMapObj.(*apiv1.ConfigMap)

	// Check if deletion has been requested.
	if baseConfigMap.DeletionTimestamp != nil {
		if err := configmapcontroller.delete(baseConfigMap); err != nil {
			glog.Errorf("Failed to delete %s: %v", configmap, err)
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "DeleteFailed",
				"ConfigMap delete failed: %v", err)
			configmapcontroller.deliverConfigMap(configmap, 0, true)
		}
		return
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for configmap: %s",
		baseConfigMap.Name)
	// Add the required finalizers before creating a configmap in underlying clusters.
	updatedConfigMapObj, err := configmapcontroller.deletionHelper.EnsureFinalizers(baseConfigMap)
	if err != nil {
		glog.Errorf("Failed to ensure delete object from underlying clusters finalizer in configmap %s: %v",
			baseConfigMap.Name, err)
		configmapcontroller.deliverConfigMap(configmap, 0, false)
		return
	}
	baseConfigMap = updatedConfigMapObj.(*apiv1.ConfigMap)

	glog.V(3).Infof("Syncing configmap %s in underlying clusters", baseConfigMap.Name)

	clusters, err := configmapcontroller.configmapFederatedInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get cluster list: %v, retrying shortly", err)
		configmapcontroller.deliverConfigMap(configmap, configmapcontroller.clusterAvailableDelay, false)
		return
	}

	operations := make([]util.FederatedOperation, 0)
	for _, cluster := range clusters {
		clusterConfigMapObj, found, err := configmapcontroller.configmapFederatedInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			glog.Errorf("Failed to get %s from %s: %v, retrying shortly", key, cluster.Name, err)
			configmapcontroller.deliverConfigMap(configmap, 0, true)
			return
		}

		// Do not modify data.
		desiredConfigMap := &apiv1.ConfigMap{
			ObjectMeta: util.DeepCopyRelevantObjectMeta(baseConfigMap.ObjectMeta),
			Data:       baseConfigMap.Data,
		}

		if !found {
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "CreateInCluster",
				"Creating configmap in cluster %s", cluster.Name)

			operations = append(operations, util.FederatedOperation{
				Type:        util.OperationTypeAdd,
				Obj:         desiredConfigMap,
				ClusterName: cluster.Name,
			})
		} else {
			clusterConfigMap := clusterConfigMapObj.(*apiv1.ConfigMap)

			// Update existing configmap, if needed.
			if !util.ConfigMapEquivalent(desiredConfigMap, clusterConfigMap) {
				configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "UpdateInCluster",
					"Updating configmap in cluster %s", cluster.Name)
				operations = append(operations, util.FederatedOperation{
					Type:        util.OperationTypeUpdate,
					Obj:         desiredConfigMap,
					ClusterName: cluster.Name,
				})
			}
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		glog.V(8).Infof("No operations needed for %s", key)
		return
	}
	err = configmapcontroller.federatedUpdater.UpdateWithOnError(operations, configmapcontroller.updateTimeout,
		func(op util.FederatedOperation, operror error) {
			configmapcontroller.eventRecorder.Eventf(baseConfigMap, api.EventTypeNormal, "UpdateInClusterFailed",
				"ConfigMap update in cluster %s failed: %v", op.ClusterName, operror)
		})

	if err != nil {
		glog.Errorf("Failed to execute updates for %s: %v, retrying shortly", key, err)
		configmapcontroller.deliverConfigMap(configmap, 0, true)
		return
	}
}

// delete deletes the given configmap or returns error if the deletion was not complete.
func (configmapcontroller *ConfigMapController) delete(configmap *apiv1.ConfigMap) error {
	glog.V(3).Infof("Handling deletion of configmap: %v", *configmap)
	_, err := configmapcontroller.deletionHelper.HandleObjectInUnderlyingClusters(configmap)
	if err != nil {
		return err
	}

	err = configmapcontroller.federatedApiClient.Core().ConfigMaps(configmap.Namespace).Delete(configmap.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of configmap finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the configmap and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete configmap: %v", err)
		}
	}
	return nil
}
